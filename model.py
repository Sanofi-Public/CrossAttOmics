from functools import reduce
from itertools import chain
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from einops.layers.torch import EinMix, Rearrange
from torch import Tensor, nn


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, p_dropout: float = 0.0):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(p_dropout)
        self.act_fn = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.act_fn(self.batch_norm(x))
        return self.dropout(x)

    @property
    def out_features(self) -> int:
        return self.linear.out_features


class FullyConnectedNetwork(nn.Module):
    def __init__(self, FC_layers: List[FullyConnectedLayer]) -> None:
        super().__init__()

        self.layers = nn.Sequential(*FC_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    @property
    def out_features(self) -> int:
        return self.layers[-1].out_features


class GroupInteraction(nn.Module):
    def __init__(
        self,
        group_size: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.attention = MultiHeadAttentionWrapper(
            embed_dim=group_size,
            num_heads=num_heads,
            k_dim=None,
            v_dim=None,
        )
        self.norm_layer = nn.LayerNorm(group_size)

    def forward(self, x: Tensor) -> Tensor:
        z = self.attention(query=x, key=x, value=x)
        z = z + x
        return self.norm_layer(z)


class IndexGroupedFCN(nn.Module):
    __constants__ = ["in_features", "out_features"]
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_group: int,
        group_size: int,
        group_spec: List[Tensor],
        proj_dim: List[List[int]],
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert len(group_spec) == n_group
        assert n_group * group_size == out_features

        self.n_group = n_group

        layer_dim = [
            [idx_grp_in.size(0)] + grp_proj_dim
            for idx_grp_in, grp_proj_dim in zip(group_spec, proj_dim)
        ]

        self.list_linear = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Linear(grp_dim[i], grp_dim[i + 1], bias=bias)
                        for i in range(0, len(grp_dim) - 1)
                    ]
                )
                for grp_dim in layer_dim
            ]
        )

        for i in range(n_group):
            self.register_buffer(f"index_group_in_{i}", group_spec[i])

    def index_group_i(self, i: int) -> Tensor:
        return self.__getattr__(f"index_group_in_{i}")

    def index_groups(self) -> Iterator[Tensor]:
        for i in range(self.n_group):
            yield self.__getattr__(f"index_group_in_{i}")

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            [
                module(x[:, idx_lst])
                for idx_lst, module in zip(self.index_groups(), self.list_linear)
            ],
            1,
        )


class AttOmicsInputLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_group: int,
        group_size: int,
        num_heads: int,
        group_spec: List[Tensor],
        group_proj_dim: List[List[int]],
        flatten_output: bool,
    ) -> None:
        super().__init__()
        self.flatten_output = flatten_output
        self.n_group = n_group
        self.group_size = group_size
        self.grouped_dim = n_group * group_size

        self.grouped_mlp = IndexGroupedFCN(
            in_features=in_features,
            out_features=self.grouped_dim,
            n_group=n_group,
            group_size=group_size,
            group_spec=group_spec,
            proj_dim=group_proj_dim,
        )
        self.interaction = GroupInteraction(group_size=group_size, num_heads=num_heads)

    def forward(self, x: Tensor) -> Tensor:
        # N: Batch size, G: number of groups, s: size of a group
        # Input dim: Nxd (d: number of input features )
        x = self.grouped_mlp(x)  # dim: Nx(G*s)
        x = x.view(-1, self.n_group, self.group_size)  # dim: NxGxs
        x = self.interaction(x)  # dim: NxGxs

        if self.flatten_output:
            x = x.view(-1, self.grouped_dim)  # dim: Nx(G*s)
        return x

    @property
    def out_features(self) -> int:
        _out_features = (self.n_group, self.group_size)
        if self.flatten_output:
            _out_features = reduce((lambda x, y: x * y), _out_features)
        return _out_features


class AttOmicsLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_group: int,
        in_group_size: int,
        out_group_size: int,
        num_heads: int,
        flatten_output: bool,
    ) -> None:
        super().__init__()
        self.flatten_output = flatten_output
        self.n_group = n_group
        self.out_group_size = out_group_size
        self.grouped_dim = n_group * out_group_size
        # Transform each group with a MLP
        self.grouped_mlp = EinMix(
            "B (G s) -> B (G ss)",
            weight_shape="G ss s",
            bias_shape="G ss",
            G=n_group,
            ss=out_group_size,
            s=in_group_size,
        )
        self.interaction = GroupInteraction(
            group_size=out_group_size, num_heads=num_heads
        )

    def forward(self, x: Tensor) -> Tensor:
        # N: Batch size, G: number of groups, s: size of a group
        # Input dim: Nxd (d: number of input features )
        x = self.grouped_mlp(x)  # dim: Nx(G*s)
        x = x.view(-1, self.n_group, self.out_group_size)  # dim: NxGxs
        x = self.interaction(x)  # dim: NxGxs

        if self.flatten_output:
            x = x.view(-1, self.grouped_dim)  # dim: Nx(G*s)
        return x

    @property
    def out_features(self) -> int:
        _out_features = (self.n_group, self.out_group_size)
        if self.flatten_output:
            _out_features = reduce((lambda x, y: x * y), _out_features)
        return _out_features


def random_grouping(
    in_features: int, proj_size: int, n_group: int
) -> Tuple[List[Tensor], str, List[List[int]]]:
    idx_in = torch.randperm(in_features, dtype=torch.long)
    chunk_sizes = (idx_in.size(0) // n_group) + (
        np.arange(n_group) < (idx_in.size(0) % n_group)
    )
    idx_in = idx_in.split(chunk_sizes.tolist(), dim=0)
    idx_in = [idx.sort().values for idx in idx_in]
    group_name = [f"Random {i}" for i in range(len(idx_in))]
    return idx_in, group_name, [[proj_size] for _ in range(n_group)]


class AttOmicsEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_group: int,
        group_size_list: List[int],
        num_heads: int,
        flatten_output: bool,
    ) -> None:
        super().__init__()
        n_layers = len(group_size_list)
        grouped_dim = [g_size * n_group for g_size in group_size_list]
        connectivity = random_grouping(
            in_features=in_features, proj_size=group_size_list[0], n_group=n_group
        )

        input_layer = AttOmicsInputLayer(
            in_features=in_features,
            n_group=n_group,
            group_size=group_size_list[0],
            group_spec=connectivity[0],
            num_heads=num_heads,
            group_proj_dim=connectivity[2],
            flatten_output=flatten_output if n_layers == 1 else True,
        )
        attOmics_layers = [input_layer]
        for i in range(1, n_layers):
            attOmics_layers.append(
                AttOmicsLayer(
                    in_features=grouped_dim[i - 1],
                    n_group=n_group,
                    in_group_size=group_size_list[i - 1],
                    out_group_size=group_size_list[i],
                    num_heads=num_heads,
                    flatten_output=flatten_output if (i == (n_layers - 1)) else True,
                )
            )
        self.attOmics_layers = nn.Sequential(*attOmics_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.attOmics_layers(x)

    @property
    def out_features(self) -> int:
        return self.attOmics_layers[-1].out_features


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        k_dim: int | None = None,
        v_dim: int | None = None,
    ) -> None:
        super().__init__()

        if k_dim is None:
            k_dim = embed_dim
        if v_dim is None:
            v_dim = embed_dim

        self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_k = nn.Linear(k_dim, embed_dim, bias=False)
        self.to_v = nn.Linear(v_dim, embed_dim, bias=False)
        self.to_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.create_heads = Rearrange("b n (h d) -> b h n d", h=num_heads)
        self.fuse_heads = Rearrange("b h n d -> b n (h d)")

        self.attention_fn = nn.functional.scaled_dot_product_attention

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        q = self.to_q(query)
        q = self.create_heads(q)

        k = self.to_k(key)
        k = self.create_heads(k)

        v = self.to_v(value)
        v = self.create_heads(v)

        x = self.attention_fn(q, k, v, dropout_p=0)
        x = self.fuse_heads(x)
        x = self.to_out(x)

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        k_dim: int,
        v_dim: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttentionWrapper(
            embed_dim=embed_dim,
            num_heads=num_heads,
            k_dim=k_dim,
            v_dim=v_dim,
        )

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        z = self.attention(
            query=target,
            key=source,
            value=source,
        )
        z = z + target
        return self.norm(z)


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttentionWrapper(
            embed_dim=embed_dim,
            num_heads=num_heads,
            k_dim=None,
            v_dim=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.attention(query=x, key=x, value=x)
        z = z + x
        return self.norm(z)


class OmicsInteraction(nn.Module):
    def __init__(
        self,
        omics: List[str],
        interaction_graph: Dict[str, List[str]],
        cross_attention_blocks: Dict[str, CrossAttentionBlock],
        self_attention_blocks: Dict[str, SelfAttentionBlock] | None = None,
        add_unimodal_branches: bool = True,
        add_unimodal_to_multimodal: bool = False,
    ) -> None:
        super().__init__()
        self.add_unimodal_branches = add_unimodal_branches
        self.add_unimodal_to_multimodal = add_unimodal_to_multimodal
        self.omics = omics
        self.interaction_graph = interaction_graph

        self.not_target_modalities = sorted(
            set(omics)
            - set(
                interaction_graph.keys()
            )  # interaction graph is a dict, keys are target modalities
        )
        self.cross_layers = nn.ModuleDict(
            cross_attention_blocks
        )  # module dict, key is a str: source-_-target

        self.use_SA = False
        if self_attention_blocks:
            self.use_SA = True
            self.sa_layers = nn.ModuleDict(
                self_attention_blocks
            )  # dict too, modality: layer

        self.flatten_group = Rearrange("b G s -> b (G s)")

    def __apply_cross_attention(self, target, source, ca_key):
        return self.cross_layers[ca_key](target=target, source=source)

    def __apply_self_attention(self, z):
        # Dict[str, Tensor[N,G,s]]
        return {omics: self.sa_layers[omics](x_target) for omics, x_target in z.items()}

    def __handle_not_a_target_modalities(self, x, z):
        if self.add_unimodal_branches:
            not_target_mod = map(x.get, self.not_target_modalities)

            z = torch.cat(
                [
                    self.flatten_group(zz)
                    for zz in chain(
                        z.values(),
                        not_target_mod,
                    )
                ],
                dim=1,
            )
        else:
            z = torch.cat([self.flatten_group(zz) for zz in z.values()], dim=1)
        return z

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        # x: Dict[str, Tensor[N,G,s]]
        z = {}
        for target, sources in self.interaction_graph.items():
            cross_att_list = []
            for source in sources:
                ca_key = source + "-_-" + target
                ca_res = self.__apply_cross_attention(x[target], x[source], ca_key)
                cross_att_list.append(ca_res)

            if self.add_unimodal_to_multimodal:
                cross_att_list.append(x[target])

            z[target] = torch.cat(cross_att_list, dim=1)

        if self.use_SA:
            z = self.__apply_self_attention(z)

        z = self.__handle_not_a_target_modalities(x, z)

        return z


class AttOmics(nn.Module):
    nice_name: str = "AttOmics"
    color: str = "#b80058"

    def __init__(
        self,
        encoder: AttOmicsEncoder,
        head: FullyConnectedNetwork,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        in_dim = head.out_features
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.head(x)
        return self.classifier(x)


class CrossAttOmics(nn.Module):
    nice_name: str = "CrossAttOmics"
    color: str = "#ebac23"

    def __init__(
        self,
        num_classes: Dict[str, int],
        modalities_encoders: Dict[str, nn.Module],
        fusion: OmicsInteraction,
        multimodal_encoder: FullyConnectedNetwork,
    ) -> None:
        super().__init__()
        self.modalities_encoders = nn.ModuleDict(modalities_encoders)
        self.fusion = fusion
        self.multimodal_encoder = multimodal_encoder
        in_dim = multimodal_encoder.out_features
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        x = {
            omics: self.modalities_encoders[omics](x_omics)
            for omics, x_omics in x.items()
        }
        x = self.fusion(x)
        x = self.multimodal_encoder(x)
        return self.classifier(x)
