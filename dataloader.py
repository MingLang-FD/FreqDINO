import os
from skimage import io
import numpy as np
from torch.utils.data import Dataset

class BinaryLoader(Dataset):
        def __init__(self, data_size, datapaths, jsfiles, transforms):
            self.path = datapaths
            self.jsfiles = jsfiles
            self.transforms = transforms
            self.img_size = data_size

        
        def __len__(self):
            return len(self.jsfiles)
              
        
        def __getitem__(self,idx):
       
            image_id = list(self.jsfiles[idx].split('.'))[0]
            image_path = os.path.join(self.path,'image/',image_id)
            mask_path = os.path.join(self.path,'mask/',image_id)


            img = io.imread(image_path+'.png')[:,:,:3].astype('float32')
            mask = io.imread(mask_path+'.png', as_gray=True)

            mask[mask>0]=255
            mask[mask<255]=0
            
            mask = mask.astype(np.uint8)

            data_group = self.transforms(image=img, mask=mask)
            img_transformed = data_group['image']
            mask = data_group['mask']

            final_img = img_transformed
            
            return (final_img, mask, image_id) 
        

