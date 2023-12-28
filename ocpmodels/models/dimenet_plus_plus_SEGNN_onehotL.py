"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

---

This code borrows heavily from the DimeNet implementation as part of
pytorch-geometric: https://github.com/rusty1s/pytorch_geometric. License:

---

Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from typing import Optional

import torch
from torch import nn
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer,
    # EmbeddingBlock,
    ResidualLayer,
    SphericalBasisLayer,
)
from torch_geometric.nn.resolver import activation_resolver
from torch_scatter import scatter
from torch_sparse import SparseTensor

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel


try:
    import sympy as sym
except ImportError:
    sym = None

#!mag
from functools import reduce
from .gemnet.layers.base_layers import Dense
from .gemnet.layers.base_layers import ResidualLayer as OCPResidualLayer
from .gemnet.layers.radial_basis import RadialBasis
from .gemnet_oc.layers.radial_basis import GaussianBasis


from torch.nn import Embedding, Linear
from typing import Callable, Dict, Optional, Tuple, Union
from torch import Tensor

# from ocpmodels.datasets.embeddings import SPIN_ONEHOT_EMBEDDINGS
from math import sqrt

# TODO
# 1.先不改变dimensionality.
# 2. 测试onehot
# 3. 测试onehot + linear
# 4. 测试learnable embedding.
def encode_to_one_hot(values):
    # Map the values to indices: -1 -> 0, 0 -> 1, 1 -> 2
    indices = values + 1

    # Number of classes for one-hot encoding (3 classes: -1, 0, 1)
    num_classes = 3

    # Perform one-hot encoding
    one_hot = torch.nn.functional.one_hot(indices, num_classes=num_classes)

    return one_hot


# #1. one-hot without linear layer.
# class EmbeddingBlock(torch.nn.Module):
#     def __init__(self, num_radial: int, hidden_channels: int, act: Callable):
#         super().__init__()
#         self.act = act

#         #wx, this should be learnable
#         self.emb = Embedding(95, hidden_channels)

#         # #wx, onehot should not be learnable
#         # self.spin_emb = torch.zeros(3, len(SPIN_ONEHOT_EMBEDDINGS[1]))
#         # for i in range(100):
#         #     self.embedding[i] = torch.tensor(embeddings[i + 1])

#         self.lin_rbf = Linear(num_radial, hidden_channels)
#         #!wx, origin: 3*hidden_channels
#         #!wx, need more layers?
#         self.lin = Linear(3 * hidden_channels+3*2, hidden_channels)

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
#         self.lin_rbf.reset_parameters()
#         self.lin.reset_parameters()

#     def forward(self, x: Tensor, s: Tensor, rbf: Tensor, i: Tensor, j: Tensor) -> Tensor:
#         #import pdb
#         #pdb.set_trace()
#         x = self.emb(x) #samples*hidden_channels, samples in a batch

#         #assume magmom value located at z
#         self.spin_emb = encode_to_one_hot(s)

#         rbf = self.act(self.lin_rbf(rbf))
#         #return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))
#         return self.act(
#                 self.lin(
#                     torch.cat([
#                         torch.cat([x[i], self.spin_emb[i]], dim=-1),
#                         torch.cat([x[j], self.spin_emb[j]], dim=-1),
#                         rbf],
#                         dim=-1)))


# 2. one-hot with linear layers.
# HP, spin_hidden_channels
class EmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        act: Callable,
        spin_hidden_channels=3,
    ):
        super().__init__()
        self.act = act

        # wx, this should be learnable
        self.emb = Embedding(95, hidden_channels)

        # #wx, onehot should not be learnable
        # self.spin_emb = torch.zeros(3, len(SPIN_ONEHOT_EMBEDDINGS[1]))
        # for i in range(100):
        #     self.embedding[i] = torch.tensor(embeddings[i + 1])
        self.spin_fc = Linear(3, spin_hidden_channels)

        self.lin_rbf = Linear(num_radial, hidden_channels)
        #!wx, origin: 3*hidden_channels
        #!wx, need more layers?
        self.lin = Linear(
            3 * hidden_channels + spin_hidden_channels * 2, hidden_channels
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()
        self.spin_fc.reset_parameters()

    def forward(
        self, x: Tensor, s: Tensor, rbf: Tensor, i: Tensor, j: Tensor
    ) -> Tensor:
        # import pdb
        # pdb.set_trace()
        x = self.emb(x)  # samples*hidden_channels, samples in a batch

        # assume magmom value located at z
        # linear + spin
        # import pdb
        # pdb.set_trace()
        self.spin_emb = encode_to_one_hot(s)
        self.spin_emb = self.spin_emb.to(dtype=self.spin_fc.weight.dtype)
        spin_feat = self.spin_fc(self.spin_emb)

        rbf = self.act(self.lin_rbf(rbf))
        # return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))
        return self.act(
            self.lin(
                torch.cat(
                    [
                        torch.cat([x[i], spin_feat[i]], dim=-1),
                        torch.cat([x[j], spin_feat[j]], dim=-1),
                        rbf,
                    ],
                    dim=-1,
                )
            )
        )


# 3. embedding
# #HP, spin_hidden_channels
# class EmbeddingBlock(torch.nn.Module):
#     def __init__(self, num_radial: int, hidden_channels: int, act: Callable, spin_hidden_channels=3):
#         super().__init__()
#         self.act = act

#         #wx, this should be learnable
#         self.emb = Embedding(95, hidden_channels)


#         self.spin_emb = Embedding(3, spin_hidden_channels)


#         self.lin_rbf = Linear(num_radial, hidden_channels)
#         #!wx, origin: 3*hidden_channels
#         #!wx, need more layers?
#         self.lin = Linear(3 * hidden_channels + spin_hidden_channels*2, hidden_channels)

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
#         self.lin_rbf.reset_parameters()
#         self.lin.reset_parameters()
#         #wx
#         self.spin_emb.weight.data.uniform_(-sqrt(3), sqrt(3))

#     def forward(self, x: Tensor, s: Tensor, rbf: Tensor, i: Tensor, j: Tensor) -> Tensor:
#         #import pdb
#         #pdb.set_trace()
#         x = self.emb(x) #samples*hidden_channels, samples in a batch

#         #assume magmom value located at z
#         #linear + spin
#         s_indices = s + 1
#         spin_feat =  self.spin_emb(s_indices)

#         rbf = self.act(self.lin_rbf(rbf))
#         #return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))
#         return self.act(
#                 self.lin(
#                     torch.cat([
#                         torch.cat([x[i], spin_feat[i]], dim=-1),
#                         torch.cat([x[j], spin_feat[j]], dim=-1),
#                         rbf],
#                         dim=-1)))


class InteractionPPBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        int_emb_size: int,
        basis_emb_size: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act="silu",
    ) -> None:
        act = activation_resolver(act)
        super(InteractionPPBlock, self).__init__()
        self.act = act

        # Transformations of Bessel and spherical basis representations.
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(
            num_spherical * num_radial, basis_emb_size, bias=False
        )
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)

        # Dense transformations of input messages.
        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets.
        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before and after skip connection.
        self.layers_before_skip = torch.nn.ModuleList(
            [
                ResidualLayer(hidden_channels, act)
                for _ in range(num_before_skip)
            ]
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList(
            [
                ResidualLayer(hidden_channels, act)
                for _ in range(num_after_skip)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        # Initial transformations.
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        #!mag rbf is eji. according to dimnet++
        # Transformation via Bessel basis.
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down-project embeddings and generate interaction triplet embeddings.
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis.
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings.
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputPPBlock(torch.nn.Module):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_emb_channels: int,
        out_channels: int,
        num_layers: int,
        act: str = "silu",
    ) -> None:
        act = activation_resolver(act)
        super(OutputPPBlock, self).__init__()
        self.act = act

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)
        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(0)

    def forward(self, x, rbf, i, num_nodes: Optional[int] = None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


#!mag introducing spin-distance edge.
class SpinDistanceEdge(torch.nn.Module):
    def __init__(
        self,
        num_radial: int,
        # hidden_channels: int,
        # out_emb_channels: int,
        # out_channels: int,
        # num_layers: int,
        num_gaussians: int,
        act: str = "silu",
        gaussian_trainable=False,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
        num_layers_rbf=2,
        num_layers_gaussian=2,
    ) -> None:
        act = activation_resolver(act)
        super(SpinDistanceEdge, self).__init__()
        self.act = act

        #!mag, set num_rbf = num_radial
        num_rbf = num_radial

        self._rbf = BesselBasisLayer(num_rbf, cutoff, envelope_exponent)

        #!mag gaussian
        self.gaussian = GaussianBasis(
            start=0.0,
            stop=5.0,
            num_gaussians=num_gaussians,
            trainable=gaussian_trainable,
        )

        self.lin_spin = nn.Linear(
            num_rbf + num_gaussians, num_radial, bias=True
        )

        self.layers_rbf = torch.nn.ModuleList(
            [ResidualLayer(num_rbf, act) for _ in range(num_layers_rbf)]
        )

        self.layers_gaussian = torch.nn.ModuleList(
            [
                ResidualLayer(num_gaussians, act)
                for _ in range(num_layers_gaussian)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):

        self._rbf.reset_parameters()

        #!guassian didn't implement reset_parameters()
        # self.gaussian.reset_parameters()

        #!mag
        glorot_orthogonal(self.lin_spin.weight, scale=2.0)

        self.lin_spin.bias.data.fill_(0)

        for res_layer in self.layers_rbf:
            res_layer.reset_parameters()

        for res_layer in self.layers_gaussian:
            res_layer.reset_parameters()

    def forward(self, dist, magft_cat, i, j):
        """_summary_

        Args:
            dist (_type_): _description_
            magft_cat (_type_): concatenated magmom features.
            i (_type_): indices
            j (_type_): indices

        Returns:
            _type_: _description_
        """

        # import pdb
        # pdb.set_trace()

        _rbf = self._rbf(dist)

        #! compute (S_i ./dot S_j) and remove 1st dimension
        _gaussian = self.gaussian(
            (magft_cat[i] * magft_cat[j]).sum(dim=1, keepdim=True)
        )

        #!mag, mlps(_rbf) and mlps(gaussian)
        for layer in self.layers_rbf:
            _rbf = layer(_rbf)

        for layer in self.layers_gaussian:
            _gaussian = layer(_gaussian)

        _gaussian = _gaussian.squeeze(1)

        #!stack |rbf, gaussian|
        rbf = torch.cat(
            (_rbf, _gaussian), dim=1
        )  # [n_edges, n_rbf(rbf+gaussian)]

        #!maybe, lin_spin:  input dim: num_rbf+num_gaussians, output dim: num_rbf
        return self.act(self.lin_spin(rbf))


class DimeNetPlusPlus(torch.nn.Module):
    r"""DimeNet++ implementation based on https://github.com/klicperajo/dimenet.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Embedding size used for interaction triplets
        basis_emb_size (int): Embedding size used in the basis transformation
        out_emb_channels(int): Embedding size used for atoms in the output block
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (function, optional): The activation funtion.
            (default: :obj:`silu`)
    """

    url = "https://github.com/klicperajo/dimenet/raw/master/pretrained"

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        int_emb_size: int,
        basis_emb_size: int,
        out_emb_channels: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: str = "silu",
        #!mag
        num_gaussians: int = 50,
        spin_hidden_channels: int = 3,
    ) -> None:
        act = activation_resolver(act)

        super(DimeNetPlusPlus, self).__init__()

        self.cutoff = cutoff

        if sym is None:
            raise ImportError("Package `sympy` could not be found.")

        self.num_blocks = num_blocks

        #!mag
        # self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.rbf = SpinDistanceEdge(
            num_radial=num_radial,
            num_gaussians=num_gaussians,  #!mag use 50 as default
            act=act,
            gaussian_trainable=False,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_layers_rbf=2,
            num_layers_gaussian=2,
        )

        self.sbf = SphericalBasisLayer(
            num_spherical, num_radial, cutoff, envelope_exponent
        )

        self.emb = EmbeddingBlock(
            num_radial, hidden_channels, act, spin_hidden_channels
        )

        self.output_blocks = torch.nn.ModuleList(
            [
                OutputPPBlock(
                    num_radial,
                    hidden_channels,
                    out_emb_channels,
                    out_channels,
                    num_output_layers,
                    act,
                )
                for _ in range(num_blocks + 1)
            ]
        )
        self.output_blocks_mag = torch.nn.ModuleList(
            [
                OutputPPBlock(
                    num_radial,
                    hidden_channels,
                    out_emb_channels,
                    out_channels,
                    num_output_layers,
                    act,
                )
                for _ in range(num_blocks + 1)
            ]
        )

        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionPPBlock(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )

        #!mag, for regressing magmom
        # self.out_block_mag = OutputPPBlock(
        #    num_radial,
        #    hidden_channels,
        #    out_emb_channels,
        #    out_channels,
        #    num_output_layers,
        #    act,
        # )

        self.reset_parameters()

    def reset_parameters(self) -> None:

        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

        #!mag
        for out_mag in self.output_blocks_mag:
            out_mag.reset_parameters()
        # self.out_block_mag.reset_parameters()

    def triplets(self, edge_index, cell_offsets, num_nodes: int):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(
            row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
        )
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()

        # Edge indices (k->j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()
        idx_ji = adj_t_row.storage.row()

        # Remove self-loop triplets d->b->d
        # Check atom as well as cell offset
        cell_offset_kji = cell_offsets[idx_kj] + cell_offsets[idx_ji]
        mask = (idx_i != idx_k) | torch.any(cell_offset_kji != 0, dim=-1)

        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
        idx_kj, idx_ji = idx_kj[mask], idx_ji[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, z, pos, batch=None):
        """ """
        raise NotImplementedError


@registry.register_model("dimenetplusplus_SEGNN_onehotL")
class DimeNetPlusPlusWrap(DimeNetPlusPlus, BaseModel):
    def __init__(
        self,
        num_atoms: int,
        bond_feat_dim: int,  # not used
        num_targets: int,
        use_pbc: bool = True,
        regress_forces: bool = True,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        int_emb_size: int = 64,
        basis_emb_size: int = 8,
        out_emb_channels: int = 256,
        num_spherical: int = 7,
        num_radial: int = 6,
        otf_graph: bool = False,
        cutoff: float = 10.0,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        #!mag
        num_gaussians: int = 50,
        spin_hidden_channels: int = 3,
    ) -> None:
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = 50

        super(DimeNetPlusPlusWrap, self).__init__(
            hidden_channels=hidden_channels,
            out_channels=num_targets,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            spin_hidden_channels=spin_hidden_channels,
        )

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        pos = data.pos
        batch = data.batch
        (
            edge_index,
            dist,
            _,
            cell_offsets,
            offsets,
            neighbors,
        ) = self.generate_graph(data)

        data.edge_index = edge_index
        data.cell_offsets = cell_offsets
        data.neighbors = neighbors
        j, i = edge_index

        _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index,
            data.cell_offsets,
            num_nodes=data.atomic_numbers.size(0),
        )

        # Calculate angles.
        pos_i = pos[idx_i].detach()
        pos_j = pos[idx_j].detach()
        if self.use_pbc:
            pos_ji, pos_kj = (
                pos[idx_j].detach() - pos_i + offsets[idx_ji],
                pos[idx_k].detach() - pos_j + offsets[idx_kj],
            )
        else:
            pos_ji, pos_kj = (
                pos[idx_j].detach() - pos_i,
                pos[idx_k].detach() - pos_j,
            )

        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)

        #!mag
        magft_cat = data.magft

        magft_cat = magft_cat.to(dist.device)

        rbf = self.rbf(dist, magft_cat, i, j)

        sbf = self.sbf(dist, angle, idx_kj)

        # data.magft[:,-1].long()
        # Embedding block.
        x = self.emb(
            data.atomic_numbers.long(), data.magft[:, -1].long(), rbf, i, j
        )
        # data.atomic_numbers.log()是很长的，需要找到对应的很长的spin

        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        M = self.output_blocks_mag[0](x, rbf, i, num_nodes=pos.size(0))
        # Think we should just have another variable that shares the energy weights #YURI
        # _M = self.out_block_mag(x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        _iter = 0
        for interaction_block, output_block, output_block_mag in zip(
            self.interaction_blocks,
            self.output_blocks[1:],
            self.output_blocks_mag[1:],
        ):
            #!mag, x referes to m_ji in certian iterations
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)

            #!mag, output_block takes m_ji and rbf return t_i, P is t_i
            P += output_block(x, rbf, i, num_nodes=pos.size(0))
            M += output_block_mag(x, rbf, i, num_nodes=pos.size(0))
            # _M += output_block(x, rbf, i, num_nodes=pos.size(0))

            #!mag, output magmom at last interaction block
            # _iter += 1
            # if _iter == self.num_blocks - 1:
            #    _M = self.out_block_mag(x, rbf, i, num_nodes=pos.size(0))

        #!mag, output energy for each system but magmom for each atom
        energy = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)

        return energy, M, x

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)

        energy, M, x = self._forward(data)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, M, x
        else:
            return energy, M, x

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
