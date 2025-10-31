import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from dataloader import BinaryLoader
import albumentations as A
from albumentations.pytorch import ToTensor
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
import argparse
import time
import pandas as pd
import cv2
import json
from tqdm import tqdm
from monai.metrics import compute_hausdorff_distance
from model import BaselineModel

def hd_score(p, y):
    tmp_hd = compute_hausdorff_distance(p, y)
    tmp_hd = torch.mean(tmp_hd)
    return tmp_hd.item()

class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU

class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth) 
        
        
        return dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='+', help='')
    parser.add_argument('--jsonfile', default='data_split.json',type=str, help='')
    parser.add_argument('--size', type=int, default=512, help='epoches')
    parser.add_argument('--model',default='outputs/wts/Dinov3_baseline_adapter_msfe_fbaa_fgbp_cross_decoder_BUSI_78.pth', type=str, help='model path')
    parser.add_argument('--use_msfe', action='store_true', help='Enable MSFE module')  
    parser.add_argument('--wavelet', type=str, default='haar', help='Wavelet type for MSFE') 
    parser.add_argument('--use_fbaa', action='store_true', help='Enable FBAA module')  
    parser.add_argument('--use_fgbp', action='store_true', help='Enable FGBP module')
    parser.add_argument('--use_cross_attn', action='store_true', help='Enable CrossAttention module')
    parser.add_argument('--decoder_type', type=str, default='simple', 
                       choices=['simple', 'boundary_guided'],
                       help='Decoder type used during training')
    args = parser.parse_args()  
    
    dataset_name = '_'.join(args.dataset)

    save_png = f'visual/{dataset_name}/{args.model}/'
    os.makedirs(save_png, exist_ok=True)

    args.jsonfile = "data_split.json"
    args.jsonfile = f'/home/datasets/{args.dataset[0]}/data_split.json'
    dataloader_path = f'/home/datasets/{args.dataset[0]}'
    with open(args.jsonfile, 'r') as f:
        df = json.load(f)

    test_files = df['test']


    test_dataset = BinaryLoader(args.size, dataloader_path, test_files, 
        A.Compose([
            A.Resize(args.size, args.size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
            ToTensor()
        ]),
    )

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1)
    

    # construct model
    model = BaselineModel(
        img_size=512, 
        pretrained=False,  
        freeze_backbone=True,
        use_msfe=args.use_msfe,
        use_fbaa=args.use_fbaa,
        use_fgbp=args.use_fgbp,
        use_cross_attn=args.use_cross_attn,
        wavelet=args.wavelet,
        decoder_type=args.decoder_type 
        )

    
    print(f"Checkpoint: {args.model}")
    
    if os.path.exists(args.model):
  
        trained_weights = torch.load(args.model, map_location='cpu')
        
      
        if list(trained_weights.keys())[0].startswith('module.'):
            print("Detected DataParallel weights, removing 'module.' prefix...")
            cleaned_weights = {k.replace('module.', ''): v for k, v in trained_weights.items()}
        else:
            cleaned_weights = trained_weights
        
     
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_weights, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    
    model = model.cuda()
    model.eval() 
    
    TestAcc = Accuracy()
    TestPrecision = Precision()
    TestDice = Dice()
    TestRecall = Recall()
    TestF1 = F1(2)
    TestIoU = IoU()

    mIoU = []
    Accuracy = []
    Precision = []
    Recall = []
    F1_score = []
    DSC = []
    FPS = []
    image_ids = []
    hd_list = []
    
    since = time.time()
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for img, mask, img_id in tqdm(test_loader):

            img = Variable(img).cuda()            
            mask = Variable(mask).cuda()

            torch.cuda.synchronize()
            start = time.time()

            if args.decoder_type == 'simple':
                mask_pred = model(img)
            elif args.decoder_type == 'boundary_guided':
                mask_pred, _ = model(img)  
        
            torch.cuda.synchronize()
            end = time.time()
            FPS.append(end-start)

            mask_pred = torch.sigmoid(mask_pred)


            mask_pred[mask_pred >= 0.5] = 1
            mask_pred[mask_pred < 0.5] = 0


            mask_draw = mask_pred.clone().detach()
            gt_draw = mask.clone().detach()
            

            IoU = TestIoU(mask_pred,mask)
            dsc = TestDice(mask_pred,mask)
            hdscore = hd_score(mask_pred,mask)

            mask_pred = mask_pred.view(-1)
            mask = mask.view(-1)


            img_id = list(img_id[0].split('.'))[0]
            mask_numpy = mask_draw.cpu().detach().float().numpy()[0][0] 
            mask_numpy[mask_numpy==1] = 255 
            
            cv2.imwrite(f'{save_png}{img_id}_m1.png',mask_numpy)

            accuracy = TestAcc(mask_pred.cpu(),mask.cpu())
            precision = TestPrecision(mask_pred.cpu(),mask.cpu())
            recall = TestRecall(mask_pred.cpu(),mask.cpu())
            f1score = TestF1(mask_pred.cpu(),mask.cpu())
            
         
            mIoU.append(IoU.item())
            DSC.append(dsc.item())
            if not np.isinf(hdscore) and not np.isnan(hdscore):
                hd_list.append(hdscore)
            Accuracy.append(accuracy.item())
            Precision.append(precision.item())
            Recall.append(recall.item())
            F1_score.append(f1score.item())
            image_ids.append(img_id)
            torch.cuda.empty_cache()
 
            
    time_elapsed = time.time() - since
    

    result_dict = {'image_id':image_ids, 'miou':mIoU, 'dice':DSC}
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(f'{save_png}results.csv',index=False)
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    F1 = 2 * np.mean(Precision) * np.mean(Recall) / (np.mean(Precision) + np.mean(Recall))
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('FPS: {:.2f}'.format(1.0/(sum(FPS)/len(FPS))))
    print('mean IoU:',round(np.mean(mIoU),4),round(np.std(mIoU),4))
    print('mean accuracy:',round(np.mean(Accuracy),4),round(np.std(Accuracy),4))
    print('mean Precision:',round(np.mean(Precision),4))
    print('mean Recall:',round(np.mean(Recall),4))
    print('mean F1:',round(np.mean(F1),4))
    print('mean HD:',round(np.mean(hd_list),4),round(np.std(hd_list),4))
    print('mean Dice:',round(np.mean(DSC),4),round(np.std(DSC),4))

