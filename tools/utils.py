import cv2
import numpy as np
import math

import torch
from sklearn.preprocessing import normalize


def showimgFrompose(img, x, y, z, landmarks, name):
    """
    Show img form pose angle
    notice:
        pose angle not euler angle!
    Parameters:
        img - cv2.img
        x - pitch
        y - yaw
        z - roll
        landmarks = face point
        name - show image name
    """

    img2 = img.copy()
    Ry = np.array([[math.cos(y), 0, -math.sin(y)],
                   [0, 1, 0],
                   [math.sin(y), 0, math.cos(y)]])

    Rx = np.array([[1, 0, 0],
                   [0, math.cos(x), -math.sin(x)],
                   [0, math.sin(x), math.cos(x)]])
    Rz = np.array([[math.cos(z), math.sin(z), 0],
                   [-math.sin(z), math.cos(z), 0],
                   [0, 0, 1]])

    normal_vector = np.array([0, 0, 1])
    normal_vector = Ry.dot(normal_vector)
    normal_vector = Rx.dot(normal_vector)
    normal_vector = Rz.dot(normal_vector)

    up_vector = np.array([0, 1, 0])
    up_vector = Ry.dot(up_vector)
    up_vector = Rx.dot(up_vector)
    up_vector = Rz.dot(up_vector)

    tan_vector = np.array([1, 0, 0])
    tan_vector = Ry.dot(tan_vector)
    tan_vector = Rx.dot(tan_vector)
    tan_vector = Rz.dot(tan_vector)

    w2ip_matrix_NewTon = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32")

    imgSize = img2.shape
    faceNorm = 0.2 * imgSize[0] * normal_vector
    faceUp = 0.2 * imgSize[0] * up_vector
    faceTan = 0.2 * imgSize[0] * tan_vector

    projNorm = w2ip_matrix_NewTon.dot(faceNorm)
    projUp = w2ip_matrix_NewTon.dot(faceUp)
    projTan = w2ip_matrix_NewTon.dot(faceTan)

    projNormEnd = (int(projNorm[0] + imgSize[0] / 2),
                   int(projNorm[1] + imgSize[1] / 2))
    projUpEnd = (int(projUp[0] + imgSize[0] / 2),
                 int(projUp[1] + imgSize[1] / 2))
    projTanEnd = (int(projTan[0] + imgSize[0] / 2),
                  int(projTan[1] + imgSize[1] / 2))

    image_fixedpt = (int(imgSize[0] / 2), int(imgSize[1] / 2))
    # cv2.line(img2, image_fixedpt, projTanEnd, (0, 255, 0), 2)
    # cv2.line(img2, image_fixedpt, projUpEnd, (255, 0, 0), 2)
    # cv2.line(img2, image_fixedpt, projNormEnd, (0, 0, 255), 2)
    for i in landmarks:
         cv2.circle(img2,np.int16(i),1,[255,0,0])
    cv2.imshow(name, cv2.resize(img2, [500, 500]))


def showimgFromeuler(img, x, y, z, landmarks, name):
    """
    Show img form euler angle
    Parameters:
        img - cv2.img
        x - pitch
        y - yaw
        z - roll
        landmarks = face point
        name - show image name
    """
    img3 = img.copy()
    y = -y
    faceNorm, faceUp = get_normal_up_from_euler_angles(x, y, z, 'xyz')
    w2ip_matrix_NewTon = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32")
    faceTan_newton = normalize(np.cross(faceUp, faceNorm, axis=0),
                               axis=0)


    imgSize = img3.shape
    faceNorm_newton = 0.5 * imgSize[0] * faceNorm
    faceUp_newton = 0.5 * imgSize[0] * faceUp
    faceTan_newton = 0.5 * imgSize[0] * faceTan_newton

    projNorm = w2ip_matrix_NewTon.dot(faceNorm_newton)
    projUp = w2ip_matrix_NewTon.dot(faceUp_newton)
    projTan = w2ip_matrix_NewTon.dot(faceTan_newton)
    projNormEnd = (int(projNorm[0] + imgSize[0] / 2),
                   int(projNorm[1] + imgSize[1] / 2))
    projUpEnd = (int(projUp[0] + imgSize[0] / 2),
                 int(projUp[1] + imgSize[1] / 2))
    projTanEnd = (int(projTan[0] + imgSize[0] / 2),
                  int(projTan[1] + imgSize[1] / 2))

    image_fixedpt = (int(imgSize[0] / 2), int(imgSize[1] / 2))
    # cv2.line(img3, image_fixedpt, projTanEnd, (0, 255, 0), 2)  # Tangent向量为绿色
    # cv2.line(img3, image_fixedpt, projUpEnd, (255, 0, 0), 2)  # Up向量为蓝色
    # cv2.line(img3, image_fixedpt, projNormEnd, (0, 0, 255), 2)  # 面法线为红色


    # for i in landmarks:
    #      cv2.circle(img3,np.int16(i),1,[255,0,0],thickness=-1)

    cv2.imshow(name, img3)

    if cv2.waitKey(3):
        cv2.imwrite(name+".png",img3)


def get_R(x, y, z):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R


# batch*n
def normalize_vector(v, use_gpu=False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    if use_gpu:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    else:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


def compute_rotation_matrix_from_ortho6d(poses, use_gpu=True):
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw, use_gpu)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z, use_gpu)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def compute_euler_angles_from_rotation_matrices(rotation_matrices, use_gpu=True):
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6
    singular = singular.float()

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)
    zs = R[:, 1, 0] * 0

    if use_gpu:
        out_euler = torch.autograd.Variable(torch.zeros(batch, 3).cuda())
    else:
        out_euler = torch.autograd.Variable(torch.zeros(batch, 3))
    out_euler[:, 0] = x * (1 - singular) + xs * singular
    out_euler[:, 1] = y * (1 - singular) + ys * singular
    out_euler[:, 2] = z * (1 - singular) + zs * singular

    return out_euler


def get_normal_up_from_euler_angles(xAngle, yAngle, zAngle, order='xyz'):

    sx = math.sin(xAngle)
    cx = math.cos(xAngle)
    sy = math.sin(yAngle)
    cy = math.cos(yAngle)
    sz = math.sin(zAngle)
    cz = math.cos(zAngle)

    if order == 'xyz':
        euler_angles_rotation_matrix = np.array([[cy * cz, -cy * sz, sy],
                                                 [cz * sx + sy + cx * sz, cx * cz - sx * sy * sz, -cy * sx],
                                                 [-cx * cz * sy + sx * sz, cz * sx + cx * sy * sz, cx * cy]
                                                 ], dtype="float32")

    faceNormal = np.array([[0.0], [0.0], [1.0]], dtype="float32")
    faceUp = np.array([[0.0], [1.0], [0.0]], dtype="float32")
    newNormal = euler_angles_rotation_matrix.dot(faceNormal)
    newUp = euler_angles_rotation_matrix.dot(faceUp)
    return newNormal, newUp



def get_pose_angles_from_normal_up(normal, up):
    up = normalize(up, axis=0)
    normal = normalize(normal, axis=0)
    pitch_angle = -math.asin(normal[1])
    yaw_angle = math.asin(normal[0] / math.sqrt(normal[0] * normal[0] + normal[2] * normal[2]))
    cx = math.cos(pitch_angle)
    sx = math.sin(pitch_angle)
    cy = math.cos(yaw_angle)
    sy = math.sin(yaw_angle)
    euler_angles_Ryxz_matrix = np.array([[cy, sx * sy, cx * sy],
                                         [0.0, cx, -sx],
                                         [-sy, cy * sx, cx * cy]
                                         ], dtype="float32")
    X = np.array([[1.0], [0.0], [0.0]], dtype="float32")
    Y = np.array([[0.0], [1.0], [0.0]], dtype="float32")
    newX = euler_angles_Ryxz_matrix.dot(X).reshape(1, 3)
    newY = euler_angles_Ryxz_matrix.dot(Y).reshape(1, 3)
    cos_up_newX = newX.dot(up)
    cos_up_newY = newY.dot(up)
    if cos_up_newY > 1.0:
        cos_up_newY = 1.0
    elif cos_up_newY < -1.0:
        cos_up_newY = -1.0
    roll_angle = math.acos(cos_up_newY)
    if cos_up_newX > 0:
        roll_angle = -roll_angle

    return np.asarray([pitch_angle, yaw_angle, roll_angle], dtype=np.float32)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
