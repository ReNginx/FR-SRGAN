import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler

torch.backends.cudnn.benchmark = True
import matplotlib.pyplot as plt
import numpy as np
import FRVSR
import Dataset


def load_model(model_name, width, height):
    if model_name == '':
        return FRVSR.FRVSR(4, lr_height=height, lr_width=width)
    else:
        raise NotImplementedError


def run():
    # Parameters
    num_epochs = 20
    output_period = 2
    batch_size = 4
    width, height = 112, 64

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model('', width, height)
    model = model.to(device)

    train_loader, val_loader = Dataset.get_data_loaders(batch_size)
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)

    flow_criterion = nn.MSELoss().to(device)
    content_criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epoch = 1
    while epoch <= num_epochs:
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
            model.train()

        for batch_num, (lr_imgs, hr_imgs) in enumerate(train_loader, 1):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            # print(f'hrimgs.shape is {hr_imgs.shape}')
            # print(f'lrimgs.shape is {lr_imgs.shape}')
            optimizer.zero_grad()
            model.init_hidden(device)
            loss = 0

            for lr_img, hr_img in zip(lr_imgs, hr_imgs):
                # print(lr_img.shape)
                hr_est, lr_est = model(lr_img)
                content_loss = content_criterion(hr_est, hr_img)
                flow_loss = flow_criterion(lr_est, lr_img)
                print(f'content_loss is {content_loss}, flow_loss is {flow_loss}')
                loss += content_loss + flow_loss

            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f' % (
                    epoch, batch_num * 1.0 / num_train_batches,
                    running_loss / output_period
                ))
                running_loss = 0.0
                gc.collect()

        gc.collect()
        # save after every epoch
        torch.save(model.state_dict(), "models/FRVSR.%d" % epoch)

        # model.eval()

        # a helper function to calc topk error
        # def calcTopKError(loader, k, name):
        #     epoch_topk_err = 0.0
        #
        #     for batch_num, (inputs, labels) in enumerate(loader, 1):
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)
        #         outputs = model(inputs)
        #
        #         _,cls = torch.topk(outputs,dim=1,k=k)
        #         batch_topk_err = (1 - (cls.numel()-torch.nonzero(cls-labels.view(-1,1)).shape[0])/labels.numel())
        #         epoch_topk_err = epoch_topk_err * ((batch_num-1) / batch_num) \
        #                         + batch_topk_err / batch_num
        #
        #         if batch_num % output_period == 0:
        #             # print('[%d:%.2f] %s_Topk_error: %.3f' % (
        #             #     epoch,
        #             #     batch_num*1.0/num_val_batches,
        #             #     name,
        #             #     epoch_topk_err/batch_num
        #             # ))
        #             gc.collect()
        #
        #
        #     return epoch_topk_err

        gc.collect()
        epoch += 1


if __name__ == "__main__":
    print('Starting training')
    run()
    print('Training terminated')
