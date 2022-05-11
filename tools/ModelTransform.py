import torch

state_dict = torch.load("./checkpoint_epoch.pth.tar")
torch.save(state_dict,"./output/output.tar",_use_new_zipfile_serialization=False)