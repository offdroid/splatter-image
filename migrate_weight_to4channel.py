import torch
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()
device="cpu"
checkpoint = torch.load(os.path.join(args.path, "model_latest.pth"), map_location=device)
w = checkpoint["model_state_dict"]["network_with_offset.encoder.enc.64x64_conv.weight"]

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

kernel = 3
w2 = weight_init([128, 4, 3, 3], "xavier_uniform", 4*kernel*kernel, 128*kernel*kernel)
w2[:,:3,...] = w

checkpoint["model_state_dict"]["network_with_offset.encoder.enc.64x64_conv.weight"] = w2
torch.save(checkpoint, "converted.pth")
