import torch
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path-from")
parser.add_argument("--path-dino")
parser.add_argument("--path-to")
parser.add_argument("--key-dim")
args = parser.parse_args()
device = "cpu"
c_from = torch.load(
    args.path_from, map_location=device
)
c_dino = torch.load(
    args.path_dino, map_location=device
)

for k in c_from["model_state_dict"].keys():
    c_dino["model_state_dict"][k] = c_from["model_state_dict"][k]
    #print(k)

for k in c_dino["model_state_dict"].keys():
    pass
    #print(k)

keys = ["encoder.dec.8x8_in0.norm0.weight", "encoder.dec.8x8_in0.norm0.bias", "encoder.dec.8x8_in0.conv0.weight"]
for k in keys:
    k = "network_with_offset." + k
    w_from = c_from["model_state_dict"][k]
    w_dino = c_dino["model_state_dict"][k]
    v = torch.tensor(w_dino.shape)
    idx = (v == torch.full_like(v, int(args.key_dim))).nonzero(as_tuple=True)[0][0]
    idx = idx.item()
    if idx == 0:
        w_dino[:256] = w_from
    elif idx == 1:
        w_dino[:, :256] = w_from
    else:
        raise RuntimeError()
    print(k, w_dino.shape)
    c_dino["model_state_dict"][k] = w_dino

torch.save(c_dino, args.path_to)
