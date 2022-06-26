import torch

state_dict = torch.load("../checkpoint/FPA_all/snapshot/checkpoint_epoch_62.pth.tar")
torch.save(state_dict,"../output/output.tar",_use_new_zipfile_serialization=False)