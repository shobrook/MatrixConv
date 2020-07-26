# MatrixConv

`MatrixConv` is a PyTorch implementation of a graph convolutional operator that uses a CNN to update node embeddings. This is useful for graph datasets where each node represents a matrix or image, such as a [scene graph](https://en.wikipedia.org/wiki/Scene_graph).

## Installation

This module can be installed with `pip`:

```bash
$ pip install matrix_conv
```

## Usage

`MatrixConv` is built on PyTorch Geometric and derives from the [`MessagePassing`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.message_passing.MessagePassing) module. It expects an input graph where each node's "features" is a matrix. `MatrixConv`, similarly to [`NNConv`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.NNConv), also incorporates any available edge features when collecting messages from a node's neighbors.

```python
import torch
from matrix_conv import MatrixConv

# Convolutional layer
conv_layer = MatrixConv(
    in_channels=1,
    out_channels=10,
    matrix_dims=[5, 5, 5],
    num_edge_attr=3,
    kernel_dims=[2, 3, 3]
)

# Your input graph (see: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs)
x = torch.randn((3, 1, 5, 5, 5), dtype=torch.float) # Shape is [num_nodes, in_channels, *matrix_dims]
edge_index = torch.tensor([
    [0, 1, 1, 2],
    [1, 0, 2, 1]
], dtype=torch.long)
edge_attr = torch.randn((4, 3), dtype=torch.float)

# Your output graph
x = conv_layer(x, edge_index, edge_attr) # Shape is now [3, 10, 4, 3, 3]
```
