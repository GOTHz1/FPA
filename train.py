import argparse
import logging
import math
import os

import numpy as np
import torch

from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from dataset.datasets import Datasets

from model.model import AngleInference, LandmarkNet
from tools import utils
from model.loss import MultitaskingLoss

from tools.utils import AverageMeter, compute_euler_angles_from_rotation_matrices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def train(train_loader, angle_net, landmark_net, criterion, optimizer):
    losses_angle = AverageMeter()
    losses_point = AverageMeter()
    losses_angle_train = AverageMeter()

    weighted_loss, l2_point_loss, angle_loss, train_angleMae = None, None, None, None
    for img, landmark_gt, euler_gt, mat_gt in train_loader:
        img = img.to(device)
        landmark_gt = landmark_gt.to(device)
        mat_gt = mat_gt.to(device)
        euler_gt = euler_gt.to(device)
        angle_net = angle_net.to(device)
        landmark_net = landmark_net.to(device)
        mat6, features = angle_net(img)
        landmarks = landmark_net(features)
        mat = utils.compute_rotation_matrix_from_ortho6d(mat6)
        weighted_loss, l2_point_loss, angle_loss, train_angleMae = criterion(landmark_gt, landmarks, euler_gt, mat,
                                                                             mat_gt)

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        losses_point.update(l2_point_loss.item())

        losses_angle.update(angle_loss.item())
        losses_angle_train.update(train_angleMae.item())
    return weighted_loss, l2_point_loss, angle_loss, train_angleMae


def validate_test(biwi_val_dataloader, angle_net, landmark_net, criterion, output_size):
    angle_net.eval()
    landmark_net.eval()

    mne_val = []
    with torch.no_grad():
        for img, landmark_gt, pose_angle_gt, R_gt in biwi_val_dataloader:
            img = img.to(device)
            pose_angle_gt = pose_angle_gt.to(device)
            angle_net = angle_net.to(device)
            angleORmat, _ = angle_net(img)
            mat = utils.compute_rotation_matrix_from_ortho6d(angleORmat)
            angle = utils.compute_euler_angles_from_rotation_matrices(mat)
            train_angleMae = torch.mean(
                torch.sum(torch.sqrt((pose_angle_gt * 180 / math.pi - angle * 180 / math.pi) ** 2), axis=1) / 3)

            mne_val.append(train_angleMae.cpu().numpy())

    return np.mean(mne_val)


def validate(biwi_val_dataloader, angle_net, landmark_net, criterion, output_size):
    angle_net.eval()
    landmark_net.eval()
    losses_point = []
    losses_angle = []
    mne_val = []
    with torch.no_grad():
        for img, landmark_gt, pose_angle_gt, R_gt in biwi_val_dataloader:
            img = img.to(device)
            landmark_gt = landmark_gt.to(device)
            R_gt = R_gt.to(device)
            pose_angle_gt = pose_angle_gt.to(device)
            angle_net = angle_net.to(device)
            landmark_net = landmark_net.to(device)

            angleORmat, out1 = angle_net(img)
            angle = utils.compute_euler_angles_from_rotation_matrices(angleORmat)
            m = torch.bmm(R_gt, angleORmat.transpose(1, 2))  # batch*3*3
            cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
            angle_loss = torch.acos(torch.clamp(cos, -1, 1))
            train_angleMae = torch.mean(
                torch.sum(torch.sqrt((pose_angle_gt * 180 / math.pi - angle * 180 / math.pi) ** 2), axis=1) / 3)
            landmark = landmark_net(out1)
            point_loss = torch.mean(torch.sum(torch.sqrt((landmark_gt - landmark) * (landmark_gt - landmark)), axis=1))

            losses_point.append(point_loss.cpu().numpy())
            losses_angle.append(angle_loss.cpu().numpy())
            mne_val.append(train_angleMae.cpu().numpy())

    print("===> Evaluate:")
    print('Eval set: Average loss_point: {:.4f} '.format(np.mean(losses_point)))
    print('Eval set: Average loss_angle: {:.4f} '.format(np.mean(losses_angle)))
    print('Eval set: Average mne_angle: {:.4f} '.format(np.mean(mne_val)))
    return np.mean(losses_point), np.mean(losses_angle), np.mean(mne_val)


def dataset_split(full_ds, train_rate):  # full_ds为train_ds, train_rate=0.8
    train_size = int(len(full_ds) * train_rate)
    validate_size = len(full_ds) - train_size
    train_ds, validate_ds = torch.utils.data.random_split(full_ds, [train_size, validate_size])
    return train_ds, validate_ds


def main(args):
    # Step 1: parse args config
    logging.basicConfig(
        format=
        '[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)

    # Step 2: model, criterion, optimizer, scheduler
    angle_backbone = AngleInference(args.width_factor, args.input_size).to(device)
    landmark_net = LandmarkNet(args.width_factor, args.input_size, args.landmark_size).to(device)

    criterion = MultitaskingLoss()
    optimizer = torch.optim.Adam([{'params': angle_backbone.parameters()},
                                  {'params': landmark_net.parameters()}],
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.lr_patience, verbose=True)
    if args.resume:
        checkpoint = torch.load(args.resume)
        angle_backbone.load_state_dict(checkpoint["anglenet"])
        landmark_net.load_state_dict(checkpoint["landmarknet"])
        args.start_epoch = checkpoint["epoch"]

    if args.dataAUG:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.RandomErasing(1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(args.input_size)])

    wlp_300 = Datasets(args.dataroot, args.landmark_size, transform)
    aflw_test = Datasets(args.test_dataset_aflw2000, args.landmark_size, args.input_size, transform)
    biwi_test = Datasets(args.test_dataset_biwi, args.landmark_size, args.input_size, transform)
    test_aflw_dataloader = DataLoader(aflw_test,
                                      batch_size=args.val_batchsize,
                                      num_workers=args.workers,
                                      shuffle=False)
    test_biwi_dataloader = DataLoader(biwi_test,
                                      batch_size=args.val_batchsize,
                                      num_workers=args.workers,
                                      shuffle=False)

    '''
    将训练集划分为训练集和验证机
    '''
    # train_dataset, val_dataset = dataset_split(wlp_300,0.8)

    dataloader = DataLoader(wlp_300,
                            batch_size=args.train_batchsize,
                            shuffle=True,
                            num_workers=args.workers,
                            drop_last=False)

    # step 4: run
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        weighted_train_loss, l2_point_loss, train_angle_loss, train_angleMae = train(dataloader, angle_backbone,
                                                                                     landmark_net, criterion,
                                                                                     optimizer)
        filename = os.path.join(str(args.snapshot),
                                "checkpoint_epoch_" + str(epoch) + '.pth.tar')

        save_checkpoint(
            {
                'epoch': epoch,
                'anglenet': angle_backbone.state_dict(),
                'landmarknet': landmark_net.state_dict()
            }, filename)

        # '''
        ##delete
        test_aflw = validate_test(test_aflw_dataloader, angle_backbone, landmark_net,
                                  criterion, args.output_size)
        test_biwi = validate_test(test_biwi_dataloader, angle_backbone, landmark_net,
                                  criterion, args.output_size)

        scheduler.step(weighted_train_loss)
        writer.add_scalars('data/test', {
            'test_aflw': test_aflw,
            'test_biwi': test_biwi,
        }, epoch)

        # '''

        writer.add_scalar('data/lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        writer.add_scalar('data/weighted_loss', weighted_train_loss, epoch)
        writer.add_scalars('data/point_loss', {
            'train point_loss': l2_point_loss
        }, epoch)
        writer.add_scalars('data/angle_loss', {
            'train angle loss': train_angle_loss
        }, epoch)
        writer.add_scalars('data/angle_mne', {
            'train angle mne': train_angleMae
        }, epoch)

    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='FPA')

    # Training parameters
    parser.add_argument('-j', '--workers', default=0, type=int)
    parser.add_argument('--base_lr', default=0.01, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)
    parser.add_argument("--lr_patience", default=40, type=int)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=1000, type=int)
    parser.add_argument('--train_batchsize', default=256, type=int)
    parser.add_argument('--val_batchsize', default=256, type=int)
    parser.add_argument(
        '--resume',
        # default='./checkpoint/FPA/checkpoint_epoch_74.pth.tar',
        type=str,
        metavar='PATH')

    # dataSet
    parser.add_argument('--dataAUG', default=False, type=bool)
    parser.add_argument('--dataroot',
                        default='./data/300WLP/lists.txt',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--test_dataset_aflw2000',
                        default='./data/AFLW3D/lists.txt',
                        type=str)
    parser.add_argument('--test_dataset_biwi',
                        default='./data/BIWI/list.txt',
                        type=str)

    # Checkpoint
    parser.add_argument('--snapshot',
                        default='./checkpoint/FPA/snapshot/',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--log_file',
                        default="./checkpoint/FPA/train.logs",
                        type=str)
    parser.add_argument('--tensorboard',
                        default="./checkpoint/FPA/tensorboard",
                        type=str)

    # model parameter
    parser.add_argument('--width_factor',
                        default=1,
                        type=int)
    parser.add_argument('--input_size',
                        default=112,
                        type=int)
    parser.add_argument('--output_size',
                        default=6,
                        type=int)
    parser.add_argument('--landmark_size',
                        default=32,
                        type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
