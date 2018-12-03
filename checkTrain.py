import argparse

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import Dataset
import FRVSR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Single Video')
    parser.add_argument('--model', default='./models/FRVSR.3', type=str, help='generator model epoch name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = 4
    MODEL_NAME = opt.model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FRVSR.FRVSR(0, 0, 0).eval()
    model.to(device)

    # for cpu
    # model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))
    model.load_state_dict(torch.load(MODEL_NAME, device))
    model.eval()

    train_loader, val_loader = Dataset.get_data_loaders(1, False)
    lr_example, hr_example = next(iter(train_loader))

    fps = 24
    frame_numbers = 7
    # frame_numbers = 100
    lr_width = lr_example.shape[4]
    lr_height = lr_example.shape[3]
    model.set_param(batch_size=1, width=lr_width, height=lr_height)
    model.init_hidden(device)

    hr_video_size = (lr_width * UPSCALE_FACTOR,
                     lr_height * UPSCALE_FACTOR)
    lr_video_size = (lr_width, lr_height)

    output_sr_name = 'out_srf_' + str(UPSCALE_FACTOR) + '_' + 'random_sample.mp4'
    output_lr_name = 'out_srf_' + 'original' + '_' + 'random_sample.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    hr_video_writer = cv2.VideoWriter(output_sr_name, fourcc, fps, hr_video_size)
    lr_video_writer = cv2.VideoWriter(output_lr_name, fourcc, fps, lr_video_size)
    # read frame
    for image in lr_example:
        image.to(device)
        # print(f'image shape is {image.shape}')
        # if torch.cuda.is_available():
        #     image = image.cuda()

        hr_out, lr_out = model(image)


        # model.init_hidden(device)
        def output(out, writer):
            out = out.cpu()
            hr_out_img = out.data[0].numpy()
            hr_out_img *= 255.0
            hr_out_img = (np.uint8(hr_out_img)).transpose((1, 2, 0))
            # save sr video
            writer.write(hr_out_img)


        output(hr_out, hr_video_writer)
        output(lr_out, lr_video_writer)

    hr_video_writer.release()
    lr_video_writer.release()
