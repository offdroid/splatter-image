import torch
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--from-encoder")
parser.add_argument("--from-decoder")
parser.add_argument("--to")
args = parser.parse_args()
device = "cpu"
checkpoint_encoder = torch.load(
    args.from_encoder, map_location=device
)
checkpoint_decoder = torch.load(
    args.from_decoder, map_location=device
)
for k in checkpoint_encoder["model_state_dict"].keys():
    if ".dec." in k or ".out.weight" in k or ".out.bias" in k:
        checkpoint_encoder["model_state_dict"][k] = checkpoint_decoder["model_state_dict"][k]

torch.save(checkpoint_encoder, args.to)
