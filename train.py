import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import time
from torch.optim import lr_scheduler
import seaborn as sns
import pandas as pd
import argparse
from dataloader import BinaryLoader
from loss import *
from tqdm import tqdm
import json
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from model import BaselineModel

torch.set_num_threads(8)


def generate_boundary_from_mask(mask, kernel_size=5):
    """Generate pseudo boundary GT"""
    B, _, H, W = mask.shape
    boundaries = []
    
    mask_np = (mask.cpu().numpy() * 255).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    for i in range(B):
        single_mask = mask_np[i, 0]
        dilated = cv2.dilate(single_mask, kernel, iterations=1)
        eroded = cv2.erode(single_mask, kernel, iterations=1)
        boundary = dilated - eroded
        boundaries.append(boundary)
    
    boundary_tensor = torch.from_numpy(np.stack(boundaries)).unsqueeze(1).float() / 255.0
    return boundary_tensor.to(mask.device)


def train_model(model, criterion_mask, optimizer, scheduler, num_epochs=5, use_boundary_guided='simple'):
    module_str = ''
    if args.use_msfe:
        module_str = '_msfe'
        if args.use_fbaa:
            module_str += '_fbaa'
        if args.use_fgbp:
            module_str += '_fgbp'
        if args.use_cross_attn:
            module_str += '_crossattn'
        if args.decoder_type != 'simple':
            module_str += f'_{args.decoder_type}'


    since = time.time()
    
    Loss_list = {'train': [], 'valid': []}
    Accuracy_list = {'train': [], 'valid': []}
    
    best_model_wts = model.state_dict()
    lambda_boundary = 0.3
    best_mask_loss = float('inf')
    best_loss = float('inf')
    
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss_total = []
            running_loss_mask = []
            running_loss_boundary = []
            running_corrects_mask = []
            
            for img, labels, img_id in tqdm(dataloaders[phase]):
                img = Variable(img.cuda())
                labels = Variable(labels.cuda())
                
                optimizer.zero_grad()

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if use_boundary_guided == 'simple':
                        pred_mask = model(img)
                    elif use_boundary_guided == 'boundary_guided':
                        pred_mask, pred_boundary = model(img)
                    else:
                        raise ValueError(f"Unknown decoder type: {use_boundary_guided}")
                
                pred_mask = torch.sigmoid(pred_mask).float()
                labels = labels.float()
                
                loss_mask = criterion_mask(pred_mask, labels)
                score_mask1 = accuracy_metric(pred_mask, labels)

                if use_boundary_guided == 'boundary_guided':
                    gt_boundary = generate_boundary_from_mask(labels, kernel_size=5)
                    pred_boundary = torch.sigmoid(pred_boundary).float()
                    gt_boundary = gt_boundary.float()
                    
                    loss_boundary = criterion_mask(pred_boundary, gt_boundary)
                    loss_total = loss_mask + lambda_boundary * loss_boundary
                    
                    running_loss_boundary.append(loss_boundary.item())
                else:
                    loss_total = loss_mask
                
                running_loss_total.append(loss_total.item())
                running_loss_mask.append(loss_mask.item())
                running_corrects_mask.append(score_mask1.item())

                if phase == 'train':
                    loss_total.backward()
                    optimizer.step()

            epoch_loss_total = np.mean(running_loss_total)
            epoch_loss_mask = np.mean(running_loss_mask)
            epoch_acc = np.mean(running_corrects_mask)
            
            if use_boundary_guided == 'boundary_guided':
                epoch_loss_boundary = np.mean(running_loss_boundary)
                print('{} Total Loss: {:.4f} (Mask: {:.4f}, Boundary: {:.4f}) IoU: {:.4f}'.format(
                    phase, epoch_loss_total, epoch_loss_mask, epoch_loss_boundary, epoch_acc))
            else:
                print('{} Loss: {:.4f} IoU: {:.4f}'.format(
                    phase, epoch_loss_total, epoch_acc))
            
            Loss_list[phase].append(epoch_loss_total)
            Accuracy_list[phase].append(epoch_acc)

            if phase == 'valid':
                if use_boundary_guided == 'boundary_guided':
                    if epoch_loss_mask <= best_mask_loss:
                        best_mask_loss = epoch_loss_mask
                        best_model_wts = model.state_dict()
                        dataset_str = '_'.join(args.dataset)
                        
                        torch.save(best_model_wts, 
                            f'outputs/wts/Dinov3_baseline_adapter{module_str}_{dataset_str}_{epoch}.pth')
                        
                        print(f"Validation Mask Loss improved to {best_mask_loss:.4f}")
                else:
                    if epoch_loss_total <= best_loss:
                        best_loss = epoch_loss_total
                        best_model_wts = model.state_dict()
                        dataset_str = '_'.join(args.dataset)
                        
                        torch.save(best_model_wts, 
                            f'outputs/wts/Dinov3_baseline_adapter{module_str}_{dataset_str}_{epoch}.pth')
                        
                        print(f"Validation loss improved to {best_loss:.4f}")
                        
                scheduler.step()
                print(f"lr: {scheduler.get_last_lr()[0]}")
                
                if hasattr(model, 'module') and hasattr(model.module, 'fbaa'):
                    alpha_val = model.module.fbaa.alpha.item()
                    beta_val = model.module.fbaa.beta.item()
                    print(f"FBAA - Alpha: {alpha_val:.4f}, Beta: {beta_val:.4f}")
                elif hasattr(model, 'fbaa'):
                    alpha_val = model.fbaa.alpha.item()
                    beta_val = model.fbaa.beta.item()
                    print(f"FBAA - Alpha: {alpha_val:.4f}, Beta: {beta_val:.4f}")
                
                if hasattr(model, 'module') and hasattr(model.module, 'cross_attn'):
                    fusion_weight = model.module.cross_attn.fusion_weight.item()
                    print(f"CrossAttn - Fusion Weight: {fusion_weight:.4f}")
                elif hasattr(model, 'cross_attn'):
                    fusion_weight = model.cross_attn.fusion_weight.item()
                    print(f"CrossAttn - Fusion Weight: {fusion_weight:.4f}")
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    if use_boundary_guided in ['boundary_guided', 'dual_head']:
        print('{} Best val Mask Loss: {:.4f}'.format(use_boundary_guided, best_mask_loss))
    else:
        print('Best val loss: {:4f}'.format(best_loss))
    
    return Loss_list, Accuracy_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='+', default='BUSI', help='')
    parser.add_argument('--jsonfile', type=str,default='data_split.json', help='')
    parser.add_argument('--dinov3_pretrain', type=str,
                       default='/home/pretrained_pth/dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
                       help='Path to DINOv3 pretrained weights')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=300, help='number of epochs')
    parser.add_argument('--use_msfe', action='store_true', help='Enable MSFE module') 
    parser.add_argument('--wavelet', type=str, default='haar', help='Wavelet type for MSFE')  
    parser.add_argument('--use_fbaa', action='store_true', help='Enable FBAA module') 
    parser.add_argument('--use_fgbp', action='store_true', help='Enable FGBP module')
    parser.add_argument('--use_cross_attn', action='store_true', help='Enable CrossAttention module')
    parser.add_argument('--decoder_type', type=str, default='simple', 
                       choices=['simple', 'boundary_guided'],
                       help='Decoder type: simple or boundary_guided')
    args = parser.parse_args()

    os.makedirs('outputs/', exist_ok=True)
    os.makedirs('outputs/wts', exist_ok=True)
    os.makedirs('outputs/train_data', exist_ok=True)
    os.makedirs('outputs/train_image', exist_ok=True)
    os.makedirs('outputs/valid_data', exist_ok=True)
    os.makedirs('outputs/valid_image', exist_ok=True)

    args.jsonfile = f'/home/datasets/{args.dataset[0]}/data_split.json'
    dataloader_path = f'/home/datasets/{args.dataset[0]}'
    with open(args.jsonfile, 'r') as f:
        df = json.load(f)
    
    val_files = df['valid']
    train_files = df['train'] 

    train_dataset = BinaryLoader(512, dataloader_path, train_files, 
        A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
            ToTensor()
        ], additional_targets={'mask2': 'mask'}),
    )
    
    val_dataset = BinaryLoader(512, dataloader_path, val_files,
        A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ], additional_targets={'mask2': 'mask'}),
    )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)
    dataloaders = {'train':train_loader,'valid':val_loader}

    
    # construct model

    model = BaselineModel(
            img_size=512, 
            pretrained=True, 
            freeze_backbone=True,
            use_msfe=args.use_msfe, 
            use_fbaa=args.use_fbaa, 
            use_fgbp=args.use_fgbp,  
            use_cross_attn=args.use_cross_attn, 
            wavelet=args.wavelet,
            decoder_type=args.decoder_type 
            )

    model = model.cuda()

    # Loss, IoU and Optimizer
    mask_loss = BinaryMaskLoss()
    accuracy_metric = BinaryIoU()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    Loss_list, Accuracy_list = train_model(model, mask_loss, optimizer, exp_lr_scheduler,
                           num_epochs=args.epoch, use_boundary_guided=args.decoder_type)
    
    dataset_str_ = '_'.join(args.dataset)
    module_str = ''
    if args.use_msfe:
        module_str = '_msfe'
        if args.use_fbaa:
            module_str += '_fbaa'
        if args.use_fgbp:
            module_str += '_fgbp'
        if args.use_cross_attn:
            module_str += '_crossattn'
        if args.decoder_type != 'simple':
            module_str += f'_{args.decoder_type}'
    plt.title('Validation loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["valid"], 'IoU':Accuracy_list["valid"]})
    valid_data.to_csv(f'outputs/valid_data/Dinov3_baseline_adapter{module_str}_{dataset_str_}_valid_data.csv')
    
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig(f'outputs/valid_image/Dinov3_baseline_adapter{module_str}_{dataset_str_}_valid.png')
    
    plt.figure()
    plt.title('Training loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["train"],'IoU':Accuracy_list["train"]})
    valid_data.to_csv(f'outputs/train_data/Dinov3_baseline_adapter{module_str}_{dataset_str_}_train_data.csv')
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig(f'outputs/train_image/Dinov3_baseline_adapter{module_str}_{dataset_str_}_train.png')