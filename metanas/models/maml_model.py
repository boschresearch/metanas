""" MAML model
Copyright (c) 2021 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

"""

""" 
Based on https://github.com/oscarknagg/few-shot
which is licensed under MIT License,
cf. 3rd-party-licenses.txt in root directory.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.

    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class MamlModel(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        num_conv_channels: int,
        k_way: int,
        final_layer_size,
    ):
        """Simple CNN as used in MAML and Reptile


        # Arguments:
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            k_way: Number of classes the model will discriminate between
            final_layer_size: 64 for Omniglot, 1600 for miniImageNet
        """
        super(MamlModel, self).__init__()
        self.conv1 = conv_block(num_input_channels, num_conv_channels)
        self.conv2 = conv_block(num_conv_channels, num_conv_channels)
        self.conv3 = conv_block(num_conv_channels, num_conv_channels)
        self.conv4 = conv_block(num_conv_channels, num_conv_channels)

        self.logits = nn.Linear(final_layer_size, k_way)

        self.criterion = nn.CrossEntropyLoss()

        ### dummy alphas to not break code
        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(2):
            self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(1, 5)))
            self.alpha_reduce.append(nn.Parameter(1e-3 * torch.randn(1, 5)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if "alpha" in n:
                self._alphas.append((n, p))

        self.alpha_prune_threshold = 0.0

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)

        return self.logits(x)

    # some dummy function to not break code
    def weights(self):
        return self.parameters()

    def named_weights(self):
        return self.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p
        # return None

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
        # return None

    def genotype(self):
        return None

    def get_sparse_num_params(
        self, alpha_prune_threshold=0.0
    ):  # dummy function to not break code
        """Get number of parameters for sparse one-shot-model

        Returns:
            A torch tensor
        """
        return None

    def drop_path_prob(self, p):
        """  does not exists for MAML model, do nothing """

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)
