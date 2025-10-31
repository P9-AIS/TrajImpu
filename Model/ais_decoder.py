import torch
import torch.nn as nn


class AISDecoder(nn.Module):
    def __init__(self, input_dim, num_attr):
        super().__init__()
        self.input_dim = input_dim
        self.num_attr = num_attr
        self.linear = nn.Linear(input_dim, num_attr)

    def forward(self, input_data):
        x = self.linear(input_data)
        return x
