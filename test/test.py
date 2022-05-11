import argparse
import math
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset.datasets import Datasets

from model.model import AngleInference, LandmarkNet
from tools import utils

cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_nme(preds, target):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark 
    """
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i,], target[i,]
        if L == 19:  # aflw
            interocular = 34  # meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8,] - pts_gt[9,])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36,] - pts_gt[45,])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60,] - pts_gt[72,])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt,
                                        axis=1)) / (interocular * L)

    return rmse


def compute_auc(errors, failureThreshold, step=0.0001, showCurve=True):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))
    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()

    return AUC, failureRate


def compute_mae_euger(euler_pre, euler_gt):
    angle_pre = euler_pre
    angle_gt = euler_gt

    N = angle_pre.shape[0]
    L = angle_pre.shape[1]

    pitch = np.zeros(N)
    yaw = np.zeros(N)
    roll = np.zeros(N)

    for i in range(N):
        pts_pred_euler, pts_gt_euler = angle_pre[i,], angle_gt[i,]

        abs = np.abs(pts_pred_euler - pts_gt_euler)

        pitch[i] = abs[0]
        yaw[i] = abs[1]
        roll[i] = abs[2]
    return pitch, yaw, roll


def validate(biwi_val_dataloader, angle_backbone, landmark_net,args):
    pitch_list = []
    yaw_list = []
    roll_list = []

    cost_time = []
    i = 0
    with torch.no_grad():
        for img, landmark_gt, euler_gt, gt_mat in biwi_val_dataloader:
            img = img.to(device)

            i = i + 1
            print(i)
            landmark_gt = landmark_gt.to(device)
            euler_gt = euler_gt.to(device)
            angle_backbone = angle_backbone.to(device)
            landmark_net = landmark_net.to(device)

            start_time = time.time()
            mat6, features = angle_backbone(img)

            mat = utils.compute_rotation_matrix_from_ortho6d(mat6)
            angle = utils.compute_euler_angles_from_rotation_matrices(mat)
            cost_time.append(time.time() - start_time)
            landmarks = landmark_net(features)

            pitch = angle[0].cpu().numpy()
            yaw = angle[1].cpu().numpy()
            roll = angle[2].cpu().numpy()

            pitch_gt = euler_gt[0].cpu().numpy()
            yaw_gt = euler_gt[1].cpu().numpy()
            roll_gt = euler_gt[2].cpu().numpy()

            landmarks = landmarks.cpu().numpy() * args.img_size
            landmarks = landmarks.reshape(landmarks.shape[0], -1,
                                          2)
            landmarks = np.int8(landmarks)
            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1,
                                              2).cpu().numpy() * args.img_size  # landmark_gt
            landmark_gt = np.int8(landmark_gt)

            if args.show_image:
                show_img = np.array(
                    np.transpose(img[0].cpu().numpy(), (1, 2, 0)))

                show_img = (show_img * 255).astype(np.uint8)
                np.clip(show_img, 0, 255)

                show_img = show_img.copy()
                show_img_gt = show_img.copy()
                pre_landmark = landmarks[0]
                pre_landmark_gt = landmark_gt[0]
                utils.showimgFromeuler(show_img, pitch, yaw, roll, pre_landmark, 'pre')
                utils.showimgFromeuler(show_img, pitch_gt, yaw_gt, roll_gt, pre_landmark_gt, 'gt')
                cv2.waitKey(0)

            pitch, yaw, roll = compute_mae_euger([pitch, yaw, roll], [pitch_gt, yaw_gt, roll_gt])
            for item in pitch:
                pitch_list.append(item)

            for item in yaw:
                yaw_list.append(item)

            for item in roll:
                roll_list.append(item)

            # nme_temp = compute_nme(landmarks, landmark_gt)
            # for item in nme_temp:
            # nme_list.append(item)
        print('pitch_mae: {:.4f}'.format(np.mean(pitch_list)))
        print('yaw_mae: {:.4f}'.format(np.mean(yaw_list)))
        print('roll_mae: {:.4f}'.format(np.mean(roll_list)))
        # nme
        '''
        print('nme: {:.4f}'.format(np.mean(nme_list)))
        # auc and failure rate
        failureThreshold = 0.1
        auc, failure_rate = compute_auc(nme_list, failureThreshold)
        print('auc @ {:.1f} failureThreshold: {:.4f}'.format(
            failureThreshold, auc))
        print('failure_rate: {:}'.format(failure_rate))
        '''
        # inference time
        print("inference_cost_time: {0:4f}".format(np.mean(cost_time)))


def main(args):
    checkpoint = torch.load(args.model_path, map_location=device)
    angle_backbone = AngleInference(args.width_factor, args.img_size).to(device)
    angle_backbone.load_state_dict(checkpoint['anglenet'])
    angle_backbone.eval()
    landmark_net = LandmarkNet(args.width_factor, args.img_size, args.landmark_size).to(device)
    landmark_net.load_state_dict(checkpoint['landmarknet'])
    landmark_net.eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(args.img_size)])
    aflw_val_dataset = Datasets(args.test_dataset,args.landmark_size, args.img_size, transform)
    aflw_val_dataloader = DataLoader(aflw_val_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=0)
    validate(aflw_val_dataloader, angle_backbone, landmark_net,args)


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path',
                        default="./checkpoint/WLP5_8/snapshot/checkpoint_epoch_111.pth.tar",
                        type=str)
    parser.add_argument('--test_dataset',
                        default='./data/BIWI/test.txt',
                        type=str)
    parser.add_argument('--show_image', default=True, type=bool)

    parser.add_argument('--width_factor',
                        default=1,
                        type=int)
    parser.add_argument('--img_size',
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
