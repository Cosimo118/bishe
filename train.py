import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet.unet_model import UNet,Discriminator

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, RFDataset
from torch.utils.data import DataLoader, random_split

# dir_img = 'data/imgs/'
# dir_mask = 'data/masks/'
dir_rf = 'data/rf_image_data/'
dir_img = 'data/image_data/'
dir_mask = 'data/masks_512/'

dir_checkpoint = 'checkpoints/'

def train_net(g_net,
              s_net,
              d_net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    # dataset = BasicDataset(dir_img, dir_mask, img_scale)
    dataset = RFDataset(dir_rf,dir_img,dir_mask,img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = torch.optim.Adam(g_net.parameters(), lr=lr, betas=(0.5, 0.999))
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if g_net.n_classes > 1 else 'max', patience=2)

    # optimizer_snet = optim.RMSprop(s_net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)    
    # scheduler_snet = optim.lr_scheduler.ReduceLROnPlateau(optimizer_snet, 'min' if s_net.n_classes > 1 else 'max', patience=2)

    D_optimizer = torch.optim.Adam(d_net.parameters(), lr=lr, betas=(0.5, 0.999))
    # scheduler_dnet = optim.lr_scheduler.ReduceLROnPlateau(D_optimizer, 'min' if s_net.n_classes > 1 else 'max', patience=2)

    # if g_net.n_classes > 1:
    #     criterion = nn.CrossEntropyLoss()
    # else:
    #     criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    D = d_net
    loss_d = nn.BCELoss()
    loss_seg = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        g_net.train()
        d_net.train()
        s_net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:#这是进度条
            for batch in train_loader:
                rf_data = batch['rf_data']
                imgs = batch['image']
                true_masks = batch['mask']

                assert imgs.shape[1] == g_net.n_channels, \
                    f'Network has been defined with {g_net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                rf_data = rf_data.to(device=device, dtype=torch.float32)
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if g_net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                ## Train discriminator with real data
                real_stacks = torch.cat((imgs,rf_data,true_masks),1)

                d_real_decision = d_net(real_stacks)
                d_real_ = torch.ones(1).cuda(device)
                D_real_loss = loss_d(d_real_decision,d_real_)

                # Train discriminator with fake data
                rf_image_pred = g_net(rf_data)
                masks_pred = s_net(rf_data)
                fake_stacks = torch.cat((rf_image_pred,rf_data,masks_pred),1)

                d_fake_decision = d_net(fake_stacks)
                d_fake_ = torch.zeros(1).cuda(device)
                D_fake_loss = loss_d(d_fake_decision,d_fake_)
                # Back propagation
                D_loss = 0.5*D_real_loss + 0.5*D_fake_loss

                
                #train G
                g_loss = loss_d(d_fake_decision,d_real_)
                image_loss = criterion(rf_image_pred, imgs)
                segmentation_loss = loss_seg(true_masks,masks_pred)

                loss = 0.05*g_loss+image_loss+segmentation_loss

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'s_loss ': segmentation_loss.item(),'loss ': loss.item(),'g_loss ': g_loss.item(),'i_loss ': image_loss.item(),'d_loss ': D_loss.item()})
                if D_loss > 1.0:
                    D.zero_grad()
                    D_loss.backward(retain_graph=True)
                    D_optimizer.step()
                    # print("---d-----")

                D.zero_grad()
                optimizer.zero_grad()
                # optimizer_snet.zero_grad()

                loss.backward()
                # loss_seg.backward()

                nn.utils.clip_grad_value_(g_net.parameters(), 0.1)
                optimizer.step()
                # optimizer_snet.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in g_net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    # val_score = eval_net(g_net, val_loader, device)
                    # # scheduler.step(val_score)
                    # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    # if g_net.n_classes > 1:
                    #     logging.info('Validation cross entropy: {}'.format(val_score))
                    #     writer.add_scalar('Loss/test', val_score, global_step)
                    # else:
                    #     logging.info('Validation Dice Coeff: {}'.format(val_score))
                    #     writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if g_net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(g_net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()





# def train_net(g_net,
#               device,
#               epochs=5,
#               batch_size=1,
#               lr=0.001,
#               val_percent=0.1,
#               save_cp=True,
#               img_scale=0.5):

#     # dataset = BasicDataset(dir_img, dir_mask, img_scale)
#     dataset = RFDataset(dir_img,dir_mask,img_scale)
#     n_val = int(len(dataset) * val_percent)
#     n_train = len(dataset) - n_val
#     train, val = random_split(dataset, [n_train, n_val])
#     train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
#     val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

#     writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
#     global_step = 0

#     logging.info(f'''Starting training:
#         Epochs:          {epochs}
#         Batch size:      {batch_size}
#         Learning rate:   {lr}
#         Training size:   {n_train}
#         Validation size: {n_val}
#         Checkpoints:     {save_cp}
#         Device:          {device.type}
#         Images scaling:  {img_scale}
#     ''')

#     optimizer = optim.RMSprop(g_net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if g_net.n_classes > 1 else 'max', patience=2)
#     # if g_net.n_classes > 1:
#     #     criterion = nn.CrossEntropyLoss()
#     # else:
#     #     criterion = nn.BCEWithLogitsLoss()
#     criterion = nn.L1Loss()

#     for epoch in range(epochs):
#         g_net.train()

#         epoch_loss = 0
#         with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:#这是进度条
#             for batch in train_loader:
#                 imgs = batch['image']
#                 true_masks = batch['mask']
#                 assert imgs.shape[1] == g_net.n_channels, \
#                     f'Network has been defined with {g_net.n_channels} input channels, ' \
#                     f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
#                     'the images are loaded correctly.'

#                 imgs = imgs.to(device=device, dtype=torch.float32)
#                 mask_type = torch.float32 if g_net.n_classes == 1 else torch.long
#                 true_masks = true_masks.to(device=device, dtype=mask_type)

#                 masks_pred = g_net(imgs)
#                 loss = criterion(masks_pred, true_masks)
#                 epoch_loss += loss.item()
#                 writer.add_scalar('Loss/train', loss.item(), global_step)

#                 pbar.set_postfix(**{'loss (batch)': loss.item()})

#                 optimizer.zero_grad()
#                 loss.backward()
#                 nn.utils.clip_grad_value_(g_net.parameters(), 0.1)
#                 optimizer.step()

#                 pbar.update(imgs.shape[0])
#                 global_step += 1
#                 if global_step % (n_train // (10 * batch_size)) == 0:
#                     for tag, value in g_net.named_parameters():
#                         tag = tag.replace('.', '/')
#                         writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
#                         writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
#                     val_score = eval_net(g_net, val_loader, device)
#                     scheduler.step(val_score)
#                     writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

#                     if g_net.n_classes > 1:
#                         logging.info('Validation cross entropy: {}'.format(val_score))
#                         writer.add_scalar('Loss/test', val_score, global_step)
#                     else:
#                         logging.info('Validation Dice Coeff: {}'.format(val_score))
#                         writer.add_scalar('Dice/test', val_score, global_step)

#                     writer.add_images('images', imgs, global_step)
#                     if g_net.n_classes == 1:
#                         writer.add_images('masks/true', true_masks, global_step)
#                         writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

#         if save_cp:
#             try:
#                 os.mkdir(dir_checkpoint)
#                 logging.info('Created checkpoint directory')
#             except OSError:
#                 pass
#             torch.save(g_net.state_dict(),
#                        dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
#             logging.info(f'Checkpoint {epoch + 1} saved !')

#     writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    g_net = UNet(n_channels=1, n_classes=1, bilinear=True)
    s_net = UNet(n_channels=1, n_classes=1, bilinear=True)

    # 加载并固定与训练分割网络的参数
    s_net.load_state_dict(
        torch.load('segModel.pth', map_location=device)
    )
    # for p in s_net.parameters():
    #     p.requires_grad=False

    d_net = Discriminator(n_channels=3,n_classes=1)


    logging.info(f'Network:\n'
                 f'\t{g_net.n_channels} input channels\n'
                 f'\t{g_net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if g_net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        g_net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    g_net.to(device=device)
    d_net.to(device=device)
    s_net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(g_net=g_net,
                  s_net = s_net,
                  d_net = d_net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(g_net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
