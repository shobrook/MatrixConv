# Third Party
import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset, zeros
from torch_geometric.nn.conv import MessagePassing

# Standard Library
from functools import reduce
from operator import mul
from math import floor


def get_output_matrix_dims(i_dims, k_dims, conv_args):
    s = conv_args["stride"] if "stride" in conv_args else 1
    p = conv_args["padding"] if "padding" in conv_args else 0

    # Guide to convolutional arithmetic: https://arxiv.org/pdf/1603.07285.pdf
    return tuple(floor((i + 2 * p - k) / s) + 1 for i, k in zip(i_dims, k_dims))


class MatrixConv(MessagePassing):
    def __init__(self, in_channels, out_channels, matrix_dims, num_edge_attr,
                 kernel_dims, aggr="mean", root_cnn=True, bias=False, **kwargs):
        """
        """

        super(MatrixConv, self).__init__(aggr=aggr)

        self.in_channels, self.out_channels = in_channels, out_channels
        self.out_matrix_dims = get_output_matrix_dims(
            i_dims=matrix_dims,
            k_dims=kernel_dims,
            conv_args=kwargs
        )
        self.edge_nn = nn.Linear(
            num_edge_attr,
            out_channels * reduce(mul, self.out_matrix_dims)
        )
        cnn_layer = getattr(nn, f"Conv{len(matrix_dims)}d")

        self.message_conv = cnn_layer(
            in_channels,
            out_channels,
            kernel_dims,
            **kwargs
        ) # phi_m
        if root_cnn:
            self.root = cnn_layer(
                in_channels,
                out_channels,
                kernel_dims,
                **kwargs
            ) # phi_r
        else:
            self.register_parameter("root", None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(
                out_channels,
                *self.out_matrix_dims
            ))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_nn)
        reset(self.message_conv)
        if self.root:
            reset(self.root)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        size = (x.size(0), x.size(0))

        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        weight = self.edge_nn(edge_attr).view(
            -1,
            self.out_channels,
            *self.out_matrix_dims
        )
        message = weight * self.message_conv(x_j)

        return message

    def update(self, aggr_out, x):
        if self.bias is not None:
            aggr_out += self.bias

        return self.root(x) + aggr_out

    def __repr__(self):
        return "".join([
            self.__class__.__name__,
            f"({self.in_channels}, ",
            f"{self.out_channels})"
        ])
