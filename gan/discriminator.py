import torch
import itertools
from torch import nn
import math
from torch.nn import functional as F

# WARN: Code adapted from DINOv2 https://github.com/facebookresearch/dinov2


class CenterPadding(nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(
            itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1])
        )
        output = F.pad(x, pads)
        return output


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=2):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)


def setup_linear_classifier(sample_output, n, avgpool, num_classes=2):
    out_dim = create_linear_input(
        sample_output, use_n_blocks=n, use_avgpool=avgpool
    ).shape[1]
    linear_classifier = LinearClassifier(
        out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
    )
    return linear_classifier


class Discriminator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.preprocess = CenterPadding(14)
        with torch.no_grad():
            sample_output = self.dino.get_intermediate_layers(
                self.preprocess(torch.rand(2, 3, 64, 64)), return_class_token=True
            )
        self.head = setup_linear_classifier(sample_output, n=4, avgpool=True)

    def forward(self, x):
        with torch.no_grad():
            embs = self.dino.get_intermediate_layers(
                self.preprocess(x), return_class_token=True
            )
        return self.head(embs)


