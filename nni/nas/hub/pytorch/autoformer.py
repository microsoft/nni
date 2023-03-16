# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = [
    'AutoFormer', 'RelativePositionSelfAttention', 'RelativePosition2D',
]

from typing import Tuple, cast, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import nni
from nni.mutable import Categorical, MutableExpression, frozen_context, ensure_frozen, Mutable
from nni.nas.nn.pytorch import (
    ParametrizedModule, ModelSpace, MutableModule, Repeat, MutableConv2d, MutableLinear, MutableLayerNorm
)
from nni.nas.space import current_model
from nni.nas.oneshot.pytorch.supermodule.operation import MixedOperation
from nni.nas.profiler.pytorch.flops import FlopsResult
from nni.nas.profiler.pytorch.utils import MutableShape, ShapeTensor, profiler_leaf_module

from .utils.pretrained import load_pretrained_weight
from .utils.nn import DropPath


class RelativePosition2D(nn.Module):
    """The implementation of relative position encoding for 2D image feature maps.

    Used in :class:`RelativePositionSelfAttention`.
    """

    def __init__(self, head_embed_dim: int, length: int = 14) -> None:
        super().__init__()
        self.head_embed_dim = head_embed_dim
        self.length = length
        self.embeddings_table_v = nn.Parameter(torch.randn(length * 2 + 2, head_embed_dim))
        self.embeddings_table_h = nn.Parameter(torch.randn(length * 2 + 2, head_embed_dim))

        nn.init.trunc_normal_(self.embeddings_table_v, std=.02)
        nn.init.trunc_normal_(self.embeddings_table_h, std=.02)

    def forward(self, length_q, length_k):
        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        # init in the device directly, rather than move to device
        range_vec_q = torch.arange(length_q, device=self.embeddings_table_v.device)
        range_vec_k = torch.arange(length_k, device=self.embeddings_table_v.device)
        # compute the row and column distance
        length_q_sqrt = int(length_q ** 0.5)
        distance_mat_v = (
            torch.div(range_vec_k[None, :], length_q_sqrt, rounding_mode='trunc') -
            torch.div(range_vec_q[:, None], length_q_sqrt, rounding_mode='trunc')
        )
        distance_mat_h = (range_vec_k[None, :] % length_q_sqrt - range_vec_q[:, None] % length_q_sqrt)
        # clip the distance to the range of [-length, length]
        distance_mat_clipped_v = torch.clamp(distance_mat_v, - self.length, self.length)
        distance_mat_clipped_h = torch.clamp(distance_mat_h, - self.length, self.length)

        # translate the distance from [1, 2 * length + 1], 0 is for the cls token
        final_mat_v = distance_mat_clipped_v + self.length + 1
        final_mat_h = distance_mat_clipped_h + self.length + 1
        # pad the 0 which represent the cls token
        final_mat_v = F.pad(final_mat_v, (1, 0, 1, 0), "constant", 0)
        final_mat_h = F.pad(final_mat_h, (1, 0, 1, 0), "constant", 0)

        final_mat_v = final_mat_v.long()
        final_mat_h = final_mat_h.long()
        # get the embeddings with the corresponding distance
        embeddings = self.embeddings_table_v[final_mat_v] + self.embeddings_table_h[final_mat_h]

        return embeddings


@profiler_leaf_module
class RelativePositionSelfAttention(MutableModule):
    """
    This class is designed to support the `relative position <https://arxiv.org/pdf/1803.02155v2.pdf>`__ in attention.

    Different from the absolute position embedding,
    the relative position embedding encodes relative distance between input tokens and learn the pairwise relations of them.
    It is commonly calculated via a look-up table with learnable parameters,
    interacting with queries and keys in self-attention modules.

    This class is different from PyTorch's built-in ``nn.MultiheadAttention`` in:

    1. It supports relative position embedding.
    2. It only supports self attention.
    3. It uses fixed dimension for each head, rather than fixed total dimension.
    """

    def __init__(
        self,
        embed_dim: int | Categorical[int],
        num_heads: int | Categorical[int],
        head_dim: int | None = 64,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        rpe: bool = False,
        rpe_length: int = 14,
    ):
        super().__init__()

        # The self. attributes are only used for inspection.
        # The actual values are stored in the submodules.
        if current_model() is not None:
            self.embed_dim = ensure_frozen(embed_dim)
            self.num_heads = ensure_frozen(num_heads)
        else:
            self.embed_dim = embed_dim
            self.num_heads = num_heads

        # head_dim is fixed 64 in official AutoFormer. set head_dim = None to use flex head dim.
        self.head_dim = head_dim or (embed_dim // num_heads)
        self.scale = qk_scale or cast(int, head_dim) ** -0.5
        self.qkv_bias = qkv_bias

        if isinstance(head_dim, Mutable) and isinstance(num_heads, Mutable):
            raise ValueError('head_dim and num_heads can not be both mutable.')

        # Please refer to MixedMultiheadAttention for details.
        self.q = MutableLinear(cast(int, embed_dim), cast(int, head_dim) * num_heads, bias=qkv_bias)
        self.k = MutableLinear(cast(int, embed_dim), cast(int, head_dim) * num_heads, bias=qkv_bias)
        self.v = MutableLinear(cast(int, embed_dim), cast(int, head_dim) * num_heads, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = MutableLinear(cast(int, head_dim) * num_heads, cast(int, embed_dim))
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpe = rpe

        if self.rpe:
            if isinstance(head_dim, Mutable):
                raise ValueError('head_dim must be a fixed integer when rpe is True.')
            self.rel_pos_embed_k = RelativePosition2D(cast(int, head_dim), rpe_length)
            self.rel_pos_embed_v = RelativePosition2D(cast(int, head_dim), rpe_length)

    def freeze(self, sample) -> RelativePositionSelfAttention:
        new_module = cast(RelativePositionSelfAttention, super().freeze(sample))
        # Handle ad-hoc attributes.
        if isinstance(self.embed_dim, Mutable):
            assert new_module is not self
            new_module.embed_dim = self.embed_dim.freeze(sample)
        if isinstance(self.num_heads, Mutable):
            assert new_module is not self
            new_module.num_heads = self.num_heads.freeze(sample)
        if isinstance(self.head_dim, Mutable):
            assert new_module is not self
            new_module.head_dim = self.head_dim.freeze(sample)
        return new_module

    def forward(self, x):
        B, N, _ = x.shape

        # Infer one of head_dim and num_heads because one of them can be mutable.
        head_dim = -1 if isinstance(self.head_dim, Mutable) else self.head_dim
        num_heads = -1 if isinstance(self.num_heads, Mutable) else self.num_heads

        q = self.q(x).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        num_heads, head_dim = q.size(1), q.size(3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rpe:
            r_p_k = self.rel_pos_embed_k(N, N)
            attn = attn + (
                q.permute(2, 0, 1, 3).reshape(N, num_heads * B, head_dim) @ r_p_k.transpose(2, 1)
            ).transpose(1, 0).reshape(B, num_heads, N, N) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, num_heads * head_dim)

        if self.rpe:
            attn_1 = attn.permute(2, 0, 1, 3).reshape(N, B * num_heads, N)
            r_p_v = self.rel_pos_embed_v(N, N)
            # The size of attention is (B, num_heads, N, N), reshape it to (N, B*num_heads, N) and do batch matmul with
            # the relative position embedding of V (N, N, head_dim) get shape like (N, B*num_heads, head_dim). We reshape it to the
            # same size as x (B, num_heads, N, hidden_dim)

            x = x + (
                (attn_1 @ r_p_v)
                .transpose(1, 0)
                .reshape(B, num_heads, N, head_dim)
                .transpose(2, 1)
                .reshape(B, N, num_heads * head_dim)
            )

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _shape_forward(self, x: ShapeTensor) -> MutableShape:
        assert x.real_shape is not None
        return MutableShape(*x.real_shape)

    def _count_flops(self, x: tuple[MutableShape], y: tuple[MutableShape]) -> FlopsResult:
        """Count the FLOPs of :class:`RelativePositionSelfAttention`.

        RPE module is ignored in this computation.
        """
        _, N, __ = x[0]

        # Dimension working inside.
        interm_dim = self.head_dim * self.num_heads

        params = (
            3 * self.embed_dim * (interm_dim + self.qkv_bias) +    # in_proj
            # skip RPE
            interm_dim * (self.embed_dim + 1)                      # out_proj, bias always true
        )

        flops = (
            N * interm_dim * self.embed_dim * 3 +  # in_proj
            N * N * interm_dim +                   # QK^T
            N * interm_dim * N +                   # RPE (k)
            N * N * interm_dim +                   # AV
            N * interm_dim * N +                   # RPE (v)
            N * interm_dim * self.embed_dim        # out_proj
        )

        return FlopsResult(flops, params)


class TransformerEncoderLayer(nn.Module):
    """
    Multi-head attention + FC + Layer-norm + Dropout.

    Similar to PyTorch's ``nn.TransformerEncoderLayer`` but supports :class:`RelativePositionSelfAttention`.

    Parameters
    ----------
    embed_dim
        Embedding dimension.
    num_heads
        Number of attention heads.
    mlp_ratio
        Ratio of MLP hidden dim to embedding dim.
    drop_path
        Drop path rate.
    drop_rate
        Dropout rate.
    pre_norm
        Whether to apply layer norm before attention.
    **kwargs
        Other arguments for :class:`RelativePositionSelfAttention`.
    """

    def __init__(
        self,
        embed_dim: int | Categorical[int],
        num_heads: int | Categorical[int],
        mlp_ratio: int | float | Categorical[int] | Categorical[float] = 4.,
        drop_path: float = 0.,
        drop_rate: float = 0.,
        pre_norm: bool = True,
        **kwargs
    ):
        super().__init__()

        self.normalize_before = pre_norm

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = RelativePositionSelfAttention(embed_dim=embed_dim, num_heads=num_heads, **kwargs)

        self.attn_layer_norm = MutableLayerNorm(cast(int, embed_dim))
        self.ffn_layer_norm = MutableLayerNorm(cast(int, embed_dim))

        self.activation_fn = nn.GELU()

        self.dropout = nn.Dropout(drop_rate)

        self.fc1 = MutableLinear(
            cast(int, embed_dim),
            cast(int, MutableExpression.to_int(embed_dim * mlp_ratio))
        )
        self.fc2 = MutableLinear(
            cast(int, MutableExpression.to_int(embed_dim * mlp_ratio)),
            cast(int, embed_dim)
        )

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def forward(self, x):
        """
        Forward function.

        Parameters
        ----------
        x
            Input to the layer of shape ``(batch, patch_num, sample_embed_dim)``.

        Returns
        -------
        Encoded output of shape ``(batch, patch_num, sample_embed_dim)``.
        """
        residual = x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x)
        x = self.dropout(x)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)

        return x


class ClassToken(ParametrizedModule):
    """
    Concat `class token <https://arxiv.org/abs/2010.11929>`__ before patch embedding.

    Parameters
    ----------
    embed_dim
        The dimension of class token.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        return torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

    def _shape_forward(self, x: ShapeTensor) -> MutableShape:
        assert x.real_shape is not None
        shape = list(x.real_shape)
        return MutableShape(shape[0], shape[1] + 1, shape[2])


class AbsolutePositionEmbedding(ParametrizedModule):
    """Add absolute position embedding on patch embedding."""

    def __init__(self, length: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        return x + self.pos_embed

    def _shape_forward(self, x: ShapeTensor) -> MutableShape:
        assert x.real_shape is not None
        return x.real_shape


class AutoFormer(ModelSpace):
    """
    The search space that is proposed in `AutoFormer <https://arxiv.org/abs/2107.00651>`__.
    There are four searchable variables: depth, embedding dimension, heads number and MLP ratio.

    Parameters
    ----------
    search_embed_dim
        The search space of embedding dimension. Use a list to specify search range.
    search_mlp_ratio
        The search space of MLP ratio. Use a list to specify search range.
    search_num_heads
        The search space of number of heads. Use a list to specify search range.
    search_depth
        The search space of depth. Use a list to specify search range.
    img_size
        Size of input image.
    patch_size
        Size of image patch.
    in_channels
        Number of channels of the input image.
    num_labels
        Number of classes for classifier.
    qkv_bias
        Whether to use bias item in the qkv embedding.
    drop_rate
        Drop rate of the MLP projection in MSA and FFN.
    attn_drop_rate
        Drop rate of attention.
    drop_path_rate
        Drop path rate.
    pre_norm
        Whether to use pre_norm. Otherwise post_norm is used.
    global_pooling
        Whether to use global pooling to generate the image representation. Otherwise the cls_token is used.
    absolute_position
        Whether to use absolute positional embeddings.
    qk_scale
        The scaler on score map in self-attention.
    rpe
        Whether to use relative position encoding.
    """

    def __init__(
        self,
        search_embed_dim: Tuple[int, ...] = (192, 216, 240),
        search_mlp_ratio: Tuple[float, ...] = (3.0, 3.5, 4.0),
        search_num_heads: Tuple[int, ...] = (3, 4),
        search_depth: Tuple[int, ...] = (12, 13, 14),
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_labels: int = 1000,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        pre_norm: bool = True,
        global_pooling: bool = True,
        absolute_position: bool = True,
        qk_scale: float | None = None,
        rpe: bool = True,
    ):
        super().__init__()

        # define search space parameters
        embed_dim = nni.choice("embed_dim", list(search_embed_dim))
        depth = nni.choice("depth", list(search_depth))
        mlp_ratios = [nni.choice(f"mlp_ratio_{i}", list(search_mlp_ratio)) for i in range(max(search_depth))]
        num_heads = [nni.choice(f"num_head_{i}", list(search_num_heads)) for i in range(max(search_depth))]

        self.patch_embed = MutableConv2d(
            in_channels, cast(int, embed_dim),
            kernel_size=patch_size,
            stride=patch_size
        )
        self.patches_num = int((img_size // patch_size) ** 2)
        self.global_pooling = global_pooling

        self.cls_token = ClassToken(cast(int, embed_dim))
        self.pos_embed = AbsolutePositionEmbedding(self.patches_num + 1, cast(int, embed_dim)) if absolute_position else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, max(search_depth))]  # stochastic depth decay rule

        self.blocks = Repeat(
            lambda index: TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads[index],
                mlp_ratio=mlp_ratios[index],
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[index],
                rpe_length=img_size // patch_size,
                qk_scale=qk_scale,
                rpe=rpe,
                pre_norm=pre_norm
            ), depth
        )

        self.norm = MutableLayerNorm(cast(int, embed_dim)) if pre_norm else nn.Identity()
        self.head = MutableLinear(cast(int, embed_dim), num_labels) if num_labels > 0 else nn.Identity()

    @classmethod
    def extra_oneshot_hooks(cls, strategy):
        # General hooks agnostic to strategy.
        return [MixedAbsolutePositionEmbedding.mutate, MixedClassToken.mutate]

    @classmethod
    def preset(cls, name: str):
        """Get the model space config proposed in paper."""
        name = name.lower()
        assert name in ['tiny', 'small', 'base']
        if name == 'tiny':
            init_kwargs = {
                'search_embed_dim': (192, 216, 240),
                'search_mlp_ratio': (3.0, 3.5, 4.0),
                'search_num_heads': (3, 4),
                'search_depth': (12, 13, 14),
            }
        elif name == 'small':
            init_kwargs = {
                'search_embed_dim': (320, 384, 448),
                'search_mlp_ratio': (3.0, 3.5, 4.0),
                'search_num_heads': (5, 6, 7),
                'search_depth': (12, 13, 14),
            }
        elif name == 'base':
            init_kwargs = {
                'search_embed_dim': (528, 576, 624),
                'search_mlp_ratio': (3.0, 3.5, 4.0),
                'search_num_heads': (8, 9, 10),
                'search_depth': (14, 15, 16),
            }
        else:
            raise ValueError(f'Unsupported architecture with name: {name}')
        return init_kwargs

    @classmethod
    def load_pretrained_supernet(cls, name: str, download: bool = True, progress: bool = True) -> 'AutoFormer':
        """
        Load the related supernet checkpoints.

        Thanks to the weight entangling strategy that AutoFormer uses,
        AutoFormer releases a few trained supernet that allows thousands of subnets to be very well-trained.
        Under different constraints, different subnets can be found directly from the supernet, and used without any fine-tuning.

        Parameters
        ----------
        name
            Search space size, must be one of {'random-one-shot-tiny', 'random-one-shot-small', 'random-one-shot-base'}.
        download
            Whether to download supernet weights.
        progress
            Whether to display the download progress.

        Returns
        -------
        The loaded supernet.
        """
        legal = ['random-one-shot-tiny', 'random-one-shot-small', 'random-one-shot-base']
        if name not in legal:
            raise ValueError(f'Unsupported name: {name}. It should be one of {legal}.')
        name = name[16:]

        # RandomOneShot is the only supported strategy for now.
        from nni.nas.strategy import RandomOneShot
        init_kwargs = cls.preset(name)
        with frozen_context.bypass():
            model_space = cls(**init_kwargs)
        model_space = RandomOneShot().mutate_model(model_space)
        weight_file = load_pretrained_weight(f"autoformer-{name}-supernet", download=download, progress=progress)
        pretrained_weights = torch.load(weight_file)
        model_space.load_state_dict(pretrained_weights)
        return model_space

    @classmethod
    def load_searched_model(
        cls, name: str,
        pretrained: bool = False, download: bool = True, progress: bool = True
    ) -> nn.Module:
        """
        Load the searched subnet model.

        Parameters
        ----------
        name
            Search space size, must be one of {'autoformer-tiny', 'autoformer-small', 'autoformer-base'}.
        pretrained
            Whether initialized with pre-trained weights.
        download
            Whether to download supernet weights.
        progress
            Whether to display the download progress.

        Returns
        -------
        nn.Module
            The subnet model.
        """
        legal = ['autoformer-tiny', 'autoformer-small', 'autoformer-base']
        if name not in legal:
            raise ValueError(f'Unsupported name: {name}. It should be one of {legal}.')
        name = name[11:]
        init_kwargs = cls.preset(name)
        if name == 'tiny':
            mlp_ratio = [3.5, 3.5, 3.0, 3.5, 3.0, 3.0, 4.0, 4.0, 3.5, 4.0, 3.5, 4.0, 3.5] + [3.0]
            num_head = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3] + [3]
            arch: Dict[str, Any] = {
                'embed_dim': 192,
                'depth': 13
            }
            for i in range(14):
                arch[f'mlp_ratio_{i}'] = mlp_ratio[i]
                arch[f'num_head_{i}'] = num_head[i]
        elif name == 'small':
            mlp_ratio = [3.0, 3.5, 3.0, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.5, 4.0] + [3.0]
            num_head = [6, 6, 5, 7, 5, 5, 5, 6, 6, 7, 7, 6, 7] + [5]
            arch: Dict[str, Any] = {
                'embed_dim': 384,
                'depth': 13
            }
            for i in range(14):
                arch[f'mlp_ratio_{i}'] = mlp_ratio[i]
                arch[f'num_head_{i}'] = num_head[i]
        elif name == 'base':
            mlp_ratio = [3.5, 3.5, 4.0, 3.5, 4.0, 3.5, 3.5, 3.0, 4.0, 4.0, 3.0, 4.0, 3.0, 3.5] + [3.0, 3.0]
            num_head = [9, 9, 9, 9, 9, 10, 9, 9, 10, 9, 10, 9, 9, 10] + [8, 8]
            arch: Dict[str, Any] = {
                'embed_dim': 576,
                'depth': 14
            }
            for i in range(16):
                arch[f'mlp_ratio_{i}'] = mlp_ratio[i]
                arch[f'num_head_{i}'] = num_head[i]
        else:
            raise ValueError(f'Unsupported architecture with name: {name}')

        model_factory = cls.frozen_factory(arch)
        model = model_factory(**init_kwargs)

        if pretrained:
            weight_file = load_pretrained_weight(f"autoformer-{name}-subnet", download=download, progress=progress)
            pretrained_weights = torch.load(weight_file)
            model.load_state_dict(pretrained_weights)

        return model

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1).view(B, self.patches_num, -1)
        x = self.cls_token(x)
        x = self.pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)

        if self.global_pooling:
            x = torch.mean(x[:, 1:], dim=1)
        else:
            x = x[:, 0]

        x = self.head(x)

        return x


AutoformerSpace = AutoFormer


# one-shot implementations

class MixedAbsolutePositionEmbedding(MixedOperation, AbsolutePositionEmbedding):
    """ Mixed absolute position embedding add operation.

    Supported arguments are:

    - ``embed_dim``

    Prefix of pos_embed will be sliced.
    """
    bound_type = AbsolutePositionEmbedding
    argument_list = ['embed_dim']

    def super_init_argument(self, name: str, value_choice: MutableExpression):
        return max(value_choice.grid())

    def freeze_weight(self, embed_dim, **kwargs) -> Any:
        from nni.nas.oneshot.pytorch.supermodule._operation_utils import Slicable, MaybeWeighted
        embed_dim_ = MaybeWeighted(embed_dim)
        pos_embed = Slicable(self.pos_embed)[..., :embed_dim_]

        return {'pos_embed': pos_embed}

    def forward_with_args(self, embed_dim,
                          inputs: torch.Tensor) -> torch.Tensor:
        pos_embed = self.freeze_weight(embed_dim)['pos_embed']
        assert isinstance(pos_embed, torch.Tensor)

        return inputs + pos_embed


class MixedClassToken(MixedOperation, ClassToken):
    """Mixed class token concat operation.

    Supported arguments are:

    - ``embed_dim``

    Prefix of cls_token will be sliced.
    """
    bound_type = ClassToken
    argument_list = ['embed_dim']

    def super_init_argument(self, name: str, value_choice: MutableExpression):
        return max(value_choice.grid())

    def freeze_weight(self, embed_dim, **kwargs) -> Any:
        from nni.nas.oneshot.pytorch.supermodule._operation_utils import Slicable, MaybeWeighted
        embed_dim_ = MaybeWeighted(embed_dim)
        cls_token = Slicable(self.cls_token)[..., :embed_dim_]

        return {'cls_token': cls_token}

    def forward_with_args(self, embed_dim,
                          inputs: torch.Tensor) -> torch.Tensor:
        cls_token = self.freeze_weight(embed_dim)['cls_token']
        assert isinstance(cls_token, torch.Tensor)

        return torch.cat((cls_token.expand(inputs.shape[0], -1, -1), inputs), dim=1)
