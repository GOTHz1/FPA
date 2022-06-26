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
from dataset.datasets import Datasets,WFLWDatasets

from model.model import AngleInference, LandmarkNet, AngleInferenceO, LandmarkNetO
from tools import utils

cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_nme(preds, target):


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
    angle_pre = euler_pre*180/math.pi
    angle_gt = euler_gt*180/math.pi

    N = angle_pre.shape[0]
    L = angle_pre.shape[1]

    pitch = np.zeros(N)
    yaw = np.zeros(N)
    roll = np.zeros(N)

    for i in range(N):
        pts_pred_euler, pts_gt_euler = angle_pre[i,:], angle_gt[i,:]

        abs = np.abs(pts_pred_euler - pts_gt_euler)

        pitch[i] = abs[0]
        yaw[i] = abs[1]
        roll[i] = abs[2]
    return pitch, yaw, roll


def validate(biwi_val_dataloader, angle_backbone, landmark_net,args):
    pitch_list = []
    yaw_list = []
    roll_list = []
    nme_list = []

    cost_time = []
    i = 0
    with torch.no_grad():
        for img, landmark_gt, euler_gt, gt_mat in biwi_val_dataloader:
            img = img.to(device)

            i = i + 1

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
            angle = angle.cpu().numpy()
            euler_gt = euler_gt.cpu().numpy()
            pitch = angle[:,0]
            yaw = angle[:,1]
            roll = angle[:,2]

            pitch_gt = euler_gt[:,0]
            yaw_gt = euler_gt[:,1]
            roll_gt = euler_gt[:,2]

            landmarks = landmarks.cpu().numpy() * args.img_size
            landmarks = landmarks.reshape(landmarks.shape[0], -1,
                                          2)
            landmarks = np.int8(landmarks)
            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1,
                                              2).cpu().numpy() * args.img_size  # landmark_gt
            landmark_gt = np.int8(landmark_gt)

            if args.show_image and( math.fabs(roll[0]-roll_gt)>30*math.pi/180 or math.fabs(yaw[0]-yaw_gt)>30*math.pi/180 or math.fabs(pitch[0]-pitch_gt)>30*math.pi/180):
                show_img = np.array(
                    np.transpose(img[0].cpu().numpy(), (1, 2, 0)))

                show_img = (show_img * 255).astype(np.uint8)
                np.clip(show_img, 0, 255)

                show_img = show_img.copy()
                pre_landmark = landmarks[0]
                pre_landmark_gt = landmark_gt[0]
                pitch1 = pitch[0]
                yaw1 = yaw[0]
                roll1 = roll[0]
                utils.showimgFromeuler(show_img,pitch1, yaw1, roll1, pre_landmark, 'pre')
                utils.showimgFromeuler(show_img, pitch_gt[0], yaw_gt[0], roll_gt[0], pre_landmark_gt, 'gt')
                cv2.waitKey(0)

            pitch_mae, yaw_mae, roll_mae = compute_mae_euger(angle, euler_gt)
            for item in pitch_mae:
                pitch_list.append(item)

            for item in yaw_mae:
                yaw_list.append(item)

            for item in roll_mae:
                roll_list.append(item)

            nme_temp = compute_nme(landmarks, landmark_gt)

            for item in nme_temp:
                nme_list.append(item)
        print('pitch_mae: {:.4f}'.format(np.mean(pitch_list)))
        print('yaw_mae: {:.4f}'.format(np.mean(yaw_list)))
        print('roll_mae: {:.4f}'.format(np.mean(roll_list)))
        # nme

        print('nme: {:.4f}'.format(np.mean(nme_list)))
        # auc and failure rate
        failureThreshold = 0.1
        auc, failure_rate = compute_auc(nme_list, failureThreshold)
        print('auc @ {:.1f} failureThreshold: {:.4f}'.format(
            failureThreshold, auc))
        print('failure_rate: {:}'.format(failure_rate))

        # inference time
        print("inference_cost_time: {0:4f}".format(np.mean(cost_time)))


def main(args):
    checkpoint = torch.load(args.model_path, map_location=device)
    angle_backbone = AngleInferenceO(args.width_factor, args.img_size).to(device)
    angle_backbone.load_state_dict(checkpoint['anglenet'])
    angle_backbone.eval()
    landmark_net = LandmarkNetO(args.width_factor, args.img_size, args.landmark_size).to(device)
    landmark_net.load_state_dict(checkpoint['landmarknet'])
    landmark_net.eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(args.img_size)])
    test_dataset = Datasets(args.test_dataset,args.landmark_size, transform)
    test_loader = DataLoader(test_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=0)
    validate(test_loader, angle_backbone, landmark_net,args)


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path',
                        default="../checkpoint/FPA/snapshot/checkpoint.pth.tar",
                        type=str)
    parser.add_argument('--test_dataset',
                        default='../dataset/data/list.txt',
                        type=str)
    parser.add_argument('--show_image', default=False, type=bool)

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
                        # default=32,
                        default=68,
                        # default=98,
                        type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
