import torch
from torch import nn
import math
from tools import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultitaskingLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(MultitaskingLoss, self).__init__()
        self.eps = eps

    def forward(self, landmark_gt, landmarks, angle_matrix, angle_matrix_gt):
        l2_landmark = torch.sum(torch.sqrt(((landmark_gt - landmarks)) * ((landmark_gt - landmarks)) + self.eps), axis=1)
        angle_mat = utils.compute_euler_angles_from_rotation_matrices(angle_matrix)
        angle_mat_gt = utils.compute_euler_angles_from_rotation_matrices(angle_matrix_gt)
        angle_mse = torch.sum(torch.sqrt((angle_mat_gt * 180 / math.pi - angle_mat * 180 / math.pi) ** 2 + self.eps), axis=1) / 3
        m = torch.bmm(angle_matrix_gt, angle_matrix.transpose(1, 2))
        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        theta = torch.acos(torch.clamp(cos, -1 + self.eps, 1 - self.eps))
        return torch.mean(l2_landmark) + 100 * torch.mean(theta), torch.mean(l2_landmark), torch.mean(theta), torch.mean(angle_mse)
