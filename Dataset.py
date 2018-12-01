import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import os
from PIL import Image

data_dir = 'Data/vimeo_super_resolution_test/low_resolution/00001/'
batch = 4 # batch size of the data every time for training
batch_number = 100000  # number of batches, so we totally have batch_number * batch images
frame_counter = 7 # number of frames per folder
height = 64
weight = 112

base_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def image_loader(path):
    img = Image.open(path)
    img_tensor = base_transform(img)
    return img_tensor

class FRDataset(data.Dataset):

    def __init__(self,dir):
        self.file_dir = dir
        self.transform = base_transform
        #self.image_loader = image_loader()
        self.frames_set = os.listdir(self.file_dir)

    def __getitem__(self, index):

        frames = self.frames_set[index] #0266
        frame_tensor = torch.Tensor(size=(frame_counter, 3, height, weight))
        for img in frames:
            file_dir_frames = self.file_dir + frames
            #file_dir_frames = '/'.os.path.join(self.file_dir,frames)
            imgs_path = os.listdir(file_dir_frames)
            imgs_path.sort()
            i = 0
            for img in imgs_path:
                final_path = file_dir_frames + "/" + img
                #final_path = '/'.os.listdir(file_dir_frames,img)
                img_tensor = image_loader(final_path)
                frame_tensor[i] = img_tensor
                i = i + 1
        return frame_tensor

    def __len__(self):
        return len(self.frames_set)


FRData = FRDataset(data_dir)

data_loader = data.DataLoader(FRData, batch_size = batch)

#print(data_loader[0].size())

for i_batch, sample_batched in enumerate(data_loader):
       #print(sample_batched)
       permuted_data = sample_batched.permute(1,0,2,3,4)
       print(permuted_data.size())
