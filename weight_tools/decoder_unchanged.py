import torch
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--from-original")
parser.add_argument("--from-fine-tuned")
args = parser.parse_args()
device = "cpu"
checkpoint_encoder = torch.load(
    args.from_original, map_location=device
)
checkpoint_decoder = torch.load(
    args.from_fine_tuned, map_location=device
)

tmp = []
for k in checkpoint_encoder["model_state_dict"].keys():
    r = torch.all(checkpoint_encoder["model_state_dict"][k] == checkpoint_decoder["model_state_dict"][k])
    print(k, r.detach().numpy())
    if ".dec." in k:
        tmp.append(r)
print(all(tmp))
