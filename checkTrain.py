import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as func
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import Dataset
import FRVSR
from skimage import img_as_ubyte
from skimage.util import img_as_float32


def test_optic_flow(frame1, frame2):
    # im1 = img_as_ubyte(frame1)
    # im2 = img_as_ubyte(frame2)
    im1 = cv2.imread('im1.png')
    im2 = cv2.imread('im2.png')
    frame1 = img_as_float32(im1)
    frame2 = img_as_float32(im2)
    prvs = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    flow[..., 0] /= flow.shape[1] / 2
    flow[..., 1] /= flow.shape[0] / 2
    flow *= -1
    for i in range(flow.shape[0]):
        for j in range(flow.shape[1]):
            flow[i, j, 0] += (j / flow.shape[1] * 2 - 1)
            flow[i, j, 1] += (i / flow.shape[0] * 2 - 1)

    print(flow.shape)

    torch_frame1 = torch.unsqueeze(torch.tensor(frame1).permute(2, 0, 1), 0)

    # print(frame1.shape)
    # print(torch_frame1.shape)
    # print(torch_frame1)
    flow = flow.astype(np.float32, copy=False)
    est_frame2 = func.grid_sample(torch_frame1, torch.unsqueeze(torch.tensor(flow), 0))
    res_img = img_as_ubyte(est_frame2[0].permute(1, 2, 0).numpy())
    cv2.imwrite('est_frame2.png', res_img)
    # flow_len = np.expand_dims(np.sqrt((flow[...,0]**2 + flow[...,1]**2)), 2)
    # flow /= flow_len
    # print(flow)
    pass
    exit(0)
    # cv2.imshow('frame2', rgb)
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         break
    #     elif k == ord('s'):
    #         cv2.imwrite('opticalmyhsv.pgm', rgb)
    #
    # cap.release()
    # cv2.destroyAllWindows()

import math

def psnr(img1, img2):
    #print(img1.size())
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Single Video')
    parser.add_argument('--model', default='./models/FRVSR.XS2', type=str, help='generator model epoch name')
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
    output_aw_name = 'out_srf_' + 'warp' + '_' + 'random_sample.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    hr_video_writer = cv2.VideoWriter(output_sr_name, fourcc, fps, hr_video_size)
    lr_video_writer = cv2.VideoWriter(output_lr_name, fourcc, fps, lr_video_size)
    aw_video_writer = cv2.VideoWriter(output_aw_name, fourcc, fps, hr_video_size)
    # read frame
    # test_optic_flow(lr_example[0][0].permute(1,2,0).numpy(), \
    #                  lr_example[1][0].permute(1,2,0).numpy())

    out_psnr = 0

    for image, ground_truth in zip(lr_example, hr_example):
        image.to(device)
        # print(f'image shape is {image.shape}')
        # if torch.cuda.is_available():
        #     image = image.cuda()

        hr_out, lr_out = model(image)
        hr_out = hr_out.clone()
        lr_out = lr_out.clone()
        aw_out = model.afterWarp.clone()
        print((hr_out.data.numpy()-ground_truth.data.numpy()).shape)
        out_psnr = out_psnr + psnr(hr_out.data.numpy(),ground_truth.data.numpy()) / 3.0

    #print("psnr:" + str(psnr))


        #model.init_hidden(device)
        def output(out, writer):
            out = out.cpu()
            out_img = out.data[0].numpy()
            out_img *= 255.0
            out_img = (np.uint8(out_img)).transpose((1, 2, 0))
            # save sr video
            writer.write(out_img)


        output(hr_out, hr_video_writer)
        output(lr_out, lr_video_writer)
        output(aw_out, aw_video_writer)

    hr_video_writer.release()
    lr_video_writer.release()
    aw_video_writer.release()
    print("psnr:" + str(out_psnr))


