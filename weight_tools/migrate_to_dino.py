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

print("same number of elements", len(c_from["model_state_dict"].keys()) == len(c_dino["model_state_dict"].keys()))
print(set(c_from["model_state_dict"].keys()).difference(set(c_dino["model_state_dict"].keys())))
v = set(c_dino["model_state_dict"].keys()).difference(set(c_from["model_state_dict"].keys()))
for k in v:
    #print(k)
    pass

for k in c_from["model_state_dict"].keys():
    #c_dino["model_state_dict"][k] = c_from["model_state_dict"][k]
    #print(k)
    pass

print("------")
for k in c_dino["model_state_dict"].keys():
    if k in c_from["model_state_dict"]:
        if c_dino["model_state_dict"][k].shape != c_from["model_state_dict"][k].shape:
            #print("zeroed", k)
            #c_dino["model_state_dict"][k] = torch.zeros_like(c_dino["model_state_dict"][k])
            pass
            #print(k)
            #print(c_dino["model_state_dict"][k].shape, c_from["model_state_dict"][k].shape)
    else:
        if "feat" not in k:
            pass
            #print("zeroed", k)
            #c_dino["model_state_dict"][k] = torch.zeros_like(c_dino["model_state_dict"][k])

    #print(k)
#quit()

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
    print(k, w_from.shape, w_dino.shape)
    c_dino["model_state_dict"][k] = w_dino

torch.save(c_dino, args.path_to)
