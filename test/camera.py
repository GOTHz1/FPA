import argparse
import numpy as np
import cv2

import torch
import torchvision

from model.model import AngleInference, LandmarkNet
from mtcnn import MTCNN

from tools import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    detector = MTCNN()
    checkpoint = torch.load(args.model_path, map_location=device)
    angle_backbone = AngleInference(1, 112).to(device)
    angle_backbone.load_state_dict(checkpoint['anglenet'])
    angle_backbone.eval()
    angle_backbone = angle_backbone.to(device)
    landmark_net = LandmarkNet(1, 112, 32).to(device)
    landmark_net.load_state_dict(checkpoint['landmarknet'])
    landmark_net.eval()

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret: break
        faces = detector.detect_faces(img)

        bounding_boxes = []
        if (len(faces) > 0):
            bounding_boxe = faces[0]['box']
            bounding_boxes.append(bounding_boxe)

        for box in bounding_boxes:

            x1, y1, w1, h1 = (box[:4])
            # expansion face box
            eY = h1 / 10
            eX = w1 / 10

            w1 = np.int16(w1 + 2 * eX)
            h1 = np.int16(h1 + 2 * eY)

            x1 = np.int16(x1 - eX)
            y1 = np.int16(y1 - eY)

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0

            cropped = img[y1:(y1 + h1), x1:(x1 + w1)]

            input = cv2.resize(cropped, (112, 112))
            input = transform(input).unsqueeze(0).to(device)

            mat6, feature = angle_backbone(input)
            landmarks = landmark_net(feature)
            mat = utils.compute_rotation_matrix_from_ortho6d(mat6)
            euler_angle = utils.compute_euler_angles_from_rotation_matrices(mat)
            landmark_pre = landmarks.cpu().detach().numpy().reshape(-1, 2)
            landmark_pre[:, 0] = landmark_pre[:, 0] * w1 + x1
            landmark_pre[:, 1] = landmark_pre[:, 1] * h1 + y1

            pre_angle = euler_angle.cpu().detach().numpy()

            x = pre_angle[0][0]
            y = pre_angle[0][1]
            z = pre_angle[0][2]

            cv2.rectangle(img, [x1, y1], [x1 + w1, y1 + h1], [255, 0, 0])
            utils.showimgFromeuler(img, x, -y, z, landmark_pre, 'euler')
        if cv2.waitKey(0)==27:
            break


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path',
                        default="../checkpoint/FPA/snapshot/checkpoint.pth.tar",
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
