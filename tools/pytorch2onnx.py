# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import argparse

import onnxsim

from model.model import AngleInference,LandmarkNet
from torch.autograd import Variable
import torch
import onnx


parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument('--torch_model',
                    default="./angle_025_32_210.tar")
parser.add_argument('--angle_model', default="./output/angle.onnx")
parser.add_argument('--angle_model_sim',
                    help='Output ONNX model',
                    default="./output/angle-sim.onnx")


parser.add_argument('--land_model', default="./output/land.onnx")
parser.add_argument('--land_model_sim',
                    help='Output ONNX model',
                    default="./output/land-sim.onnx")
args = parser.parse_args()

print("=====> load pytorch checkpoint...")
checkpoint = torch.load(args.torch_model, map_location=torch.device('cpu'))

anglenet = AngleInference(0.25,112)
anglenet.load_state_dict(checkpoint['anglenet'])
landnet = LandmarkNet(0.25,112,32)
landnet.load_state_dict(checkpoint['landmarknet'])

print("anglenet:", anglenet)
print("landnet:", landnet)

print("=====> convert pytorch model to onnx...")
dummy_input = Variable(torch.randn(1, 3, 112, 112))
dummy_input2 = Variable(torch.randn(1,32,14,14))
input_names = ["input_1"]
output_names = ["output_1","output_2"]
output_name = ["output_1"]

torch.onnx.export(anglenet,
                  dummy_input,
                  args.angle_model,
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)
torch.onnx.export(landnet,
                  dummy_input2,
                  args.land_model,
                  verbose=True,
                  input_names=input_names,
                  output_names=output_name)

print("====> check onnx model...")
import onnx
model = onnx.load(args.angle_model)
onnx.checker.check_model(model)
model2 = onnx.load(args.land_model)
onnx.checker.check_model(model2)

print("====> Simplifying...")
model_opt,check = onnxsim.simplify(args.angle_model)
# print("model_opt", model_opt)
onnx.save(model_opt, args.angle_model_sim)

model_opt2,check2 = onnxsim.simplify(args.land_model)
# print("model_opt", model_opt)
onnx.save(model_opt2, args.land_model_sim)
print("onnx model simplify Ok!")