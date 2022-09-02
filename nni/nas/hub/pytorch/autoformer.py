# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Tuple, cast, Any, Dict, Union

import torch
import torch.nn.functional as F

import nni.nas.nn.pytorch as nn
from nni.nas import model_wrapper, basic_unit
from nni.nas.fixed import no_fixed_arch
from nni.nas.nn.pytorch.choice import ValueChoiceX
from nni.nas.oneshot.pytorch.supermodule.operation import MixedOperation
from nni.nas.oneshot.pytorch.supermodule._valuechoice_utils import traverse_all_options
from nni.nas.oneshot.pytorch.supermodule._operation_utils import Slicable as _S, MaybeWeighted as _W

from .utils.fixed import FixedFactory
from .utils.pretrained import load_pretrained_weight

try:
    TIMM_INSTALLED = True
    from timm.models.layers import trunc_normal_, DropPath
except ImportError:
    TIMM_INSTALLED = False


class RelativePosition2D(nn.Module):
    def __init__(self, head_embed_dim, length=14,) -> None:
        super().__init__()
        self.head_embed_dim = head_embed_dim
        self.legnth = length
        self.embeddings_table_v = nn.Parameter(torch.randn(length * 2 + 2, head_embed_dim))
        self.embeddings_table_h = nn.Parameter(torch.randn(length * 2 + 2, head_embed_dim))

        trunc_normal_(self.embeddings_table_v, std=.02)
        trunc_normal_(self.embeddings_table_h, std=.02)

    def forward(self, length_q, length_k):
        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        # init in the device directly, rather than move to device
        range_vec_q = torch.arange(length_q, device=self.embeddings_table_v.device)
        range_vec_k = torch.arange(length_k, device=self.embeddings_table_v.device)
        # compute the row and column distance
        length_q_sqrt = int(length_q ** 0.5)
        distance_mat_v = (range_vec_k[None, :] // length_q_sqrt - range_vec_q[:, None] // length_q_sqrt)
        distance_mat_h = (range_vec_k[None, :]  % length_q_sqrt - range_vec_q[:, None]  % length_q_sqrt)
        # clip the distance to the range of [-legnth, legnth]
        distance_mat_clipped_v = torch.clamp(distance_mat_v, - self.legnth, self.legnth)
        distance_mat_clipped_h = torch.clamp(distance_mat_h, - self.legnth, self.legnth)

        # translate the distance from [1, 2 * legnth + 1], 0 is for the cls token
        final_mat_v = distance_mat_clipped_v + self.legnth + 1
        final_mat_h = distance_mat_clipped_h + self.legnth + 1
        # pad the 0 which represent the cls token
        final_mat_v = F.pad(final_mat_v, (1, 0, 1, 0), "constant", 0)
        final_mat_h = F.pad(final_mat_h, (1, 0, 1, 0), "constant", 0)

        final_mat_v = final_mat_v.long()
        final_mat_h = final_mat_h.long()
        # get the embeddings with the corresponding distance
        embeddings = self.embeddings_table_v[final_mat_v] + self.embeddings_table_h[final_mat_h]

        return embeddings

class RelativePositionAttention(nn.Module):
    """
    This class is designed to support the relative position in attention.
    The pytorch built-in nn.MultiheadAttention() does not support relative position embedding.
    Different from the absolute position embedding, the relative position embedding considers
    encode the relative distance between input tokens and learn the pairwise relations of them.
    It is commonly calculated via a look-up table with learnable parameters interacting with queries
    and keys in self-attention modules.
    """
    def __init__(
            self, embed_dim, num_heads,
            attn_drop=0., proj_drop=0.,
            qkv_bias=False, qk_scale=None,
            rpe_length=14, rpe=False,
            head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        # head_dim is fixed 64 in official autoformer. set head_dim = None to use flex head dim.
        self.head_dim = head_dim or (embed_dim // num_heads)
        self.scale = qk_scale or head_dim ** -0.5

        # Please refer to MixedMultiheadAttention for details.
        self.q = nn.Linear(embed_dim, head_dim * num_heads, bias = qkv_bias)
        self.k = nn.Linear(embed_dim, head_dim * num_heads, bias = qkv_bias)
        self.v = nn.Linear(embed_dim, head_dim * num_heads, bias = qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpe = rpe
        if rpe:
            self.rel_pos_embed_k = RelativePosition2D(head_dim, rpe_length)
            self.rel_pos_embed_v = RelativePosition2D(head_dim, rpe_length)

    def forward(self, x):
        B, N, _ = x.shape
        head_dim = self.head_dim
        # num_heads can not get from self.num_heads directly,
        # use -1 to compute implicitly.
        num_heads = -1
        q = self.q(x).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        num_heads = q.size(1)

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
            x = x + (attn_1 @ r_p_v).transpose(1, 0).reshape(B, num_heads, N, head_dim).transpose(2, 1).reshape(B, N, num_heads * head_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    This class is designed to support the RelativePositionAttention().
    The pytorch build-in nn.TransformerEncoderLayer() does not support customed attention.
    """
    def __init__(
        self, embed_dim, num_heads, mlp_ratio: Union[int, float, nn.ValueChoice]=4.,
        qkv_bias=False, qk_scale=None, rpe=False,
        drop_rate=0., attn_drop=0., proj_drop=0., drop_path=0.,
        pre_norm=True, rpe_length=14, head_dim=64
    ):
        super().__init__()

        self.normalize_before = pre_norm
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dropout = drop_rate
        self.attn = RelativePositionAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            rpe=rpe,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            rpe_length=rpe_length,
            head_dim=head_dim
        )

        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)

        self.activation_fn = nn.GELU()

        self.fc1 = nn.Linear(
            cast(int, embed_dim),
            cast(int, nn.ValueChoice.to_int(embed_dim * mlp_ratio))
        )
        self.fc2 = nn.Linear(
            cast(int, nn.ValueChoice.to_int(embed_dim * mlp_ratio)),
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
        Args:
            x (Tensor): input to the layer of shape `(batch, patch_num , sample_embed_dim)`
        Returns:
            encoded output of shape `(batch, patch_num, sample_embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)

        return x


@basic_unit
class ClsToken(nn.Module):
    """ Concat class token with dim=embed_dim before patch embedding.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        return torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)


class MixedClsToken(MixedOperation, ClsToken):
    """ Mixed class token concat operation.

    Supported arguments are:

    - ``embed_dim``

    Prefix of cls_token will be sliced.
    """
    bound_type = ClsToken
    argument_list = ['embed_dim']

    def super_init_argument(self, name: str, value_choice: ValueChoiceX):
        return max(traverse_all_options(value_choice))

    def slice_param(self, embed_dim, **kwargs) -> Any:
        embed_dim_ = _W(embed_dim)
        cls_token = _S(self.cls_token)[..., :embed_dim_]

        return {'cls_token': cls_token}

    def forward_with_args(self, embed_dim,
                        inputs: torch.Tensor) -> torch.Tensor:
        cls_token = self.slice_param(embed_dim)['cls_token']
        assert isinstance(cls_token, torch.Tensor)

        return torch.cat((cls_token.expand(inputs.shape[0], -1, -1), inputs), dim=1)

@basic_unit
class AbsPosEmbed(nn.Module):
    """ Add absolute position embedding on patch embedding.
    """
    def __init__(self, length: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        return x + self.pos_embed


class MixedAbsPosEmbed(MixedOperation, AbsPosEmbed):
    """ Mixed absolute position embedding add operation.

    Supported arguments are:

    - ``embed_dim``

    Prefix of pos_embed will be sliced.
    """
    bound_type = AbsPosEmbed
    argument_list = ['embed_dim']

    def super_init_argument(self, name: str, value_choice: ValueChoiceX):
        return max(traverse_all_options(value_choice))

    def slice_param(self, embed_dim, **kwargs) -> Any:
        embed_dim_ = _W(embed_dim)
        pos_embed = _S(self.pos_embed)[..., :embed_dim_]

        return {'pos_embed': pos_embed}

    def forward_with_args(self,  embed_dim,
                        inputs: torch.Tensor) -> torch.Tensor:
        pos_embed = self.slice_param(embed_dim)['pos_embed']
        assert isinstance(pos_embed, torch.Tensor)

        return inputs + pos_embed


@model_wrapper
class AutoformerSpace(nn.Module):
    """
    The search space that is proposed in `Autoformer <https://arxiv.org/abs/2107.00651>`__.
    There are four searchable variables: depth, embedding dimension, heads number and MLP ratio.

    Parameters
    ----------
    search_embed_dim : list of int
        The search space of embedding dimension.
    search_mlp_ratio : list of float
        The search space of MLP ratio.
    search_num_heads : list of int
        The search space of number of heads.
    search_depth: list of int
        The search space of depth.
    img_size : int
        Size of input image.
    patch_size : int
        Size of image patch.
    in_chans : int
        Number of channels of the input image.
    num_classes : int
        Number of classes for classifier.
    qkv_bias : bool
        Whether to use bias item in the qkv embedding.
    drop_rate : float
        Drop rate of the MLP projection in MSA and FFN.
    attn_drop_rate : float
        Drop rate of attention.
    drop_path_rate : float
        Drop path rate.
    pre_norm : bool
        Whether to use pre_norm. Otherwise post_norm is used.
    global_pool : bool
        Whether to use global pooling to generate the image representation. Otherwise the cls_token is used.
    abs_pos : bool
        Whether to use absolute positional embeddings.
    qk_scale : float
        The scaler on score map in self-attention.
    rpe : bool
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
        in_chans: int = 3,
        num_classes: int = 1000,
        qkv_bias: bool = False,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        pre_norm: bool = True,
        global_pool: bool = False,
        abs_pos: bool = True,
        qk_scale: Optional[float] = None,
        rpe: bool = True,
    ):
        super().__init__()

        if not TIMM_INSTALLED:
            raise ImportError('timm must be installed to use AutoFormer.')

        # define search space parameters
        embed_dim = nn.ValueChoice(list(search_embed_dim), label="embed_dim")
        depth = nn.ValueChoice(list(search_depth), label="depth")
        mlp_ratios = [nn.ValueChoice(list(search_mlp_ratio), label=f"mlp_ratio_{i}") for i in range(max(search_depth))]
        num_heads = [nn.ValueChoice(list(search_num_heads), label=f"num_head_{i}") for i in range(max(search_depth))]

        self.patch_embed = nn.Conv2d(
            in_chans, cast(int, embed_dim),
            kernel_size = patch_size,
            stride = patch_size
        )
        self.patches_num = int((img_size // patch_size) ** 2)
        self.global_pool = global_pool

        self.cls_token = ClsToken(cast(int, embed_dim))
        self.pos_embed = AbsPosEmbed(self.patches_num+1, cast(int, embed_dim)) if abs_pos else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, max(search_depth))]  # stochastic depth decay rule

        self.blocks = nn.Repeat(
            lambda index: TransformerEncoderLayer(
                embed_dim = embed_dim, num_heads = num_heads[index], mlp_ratio=mlp_ratios[index],
                qkv_bias = qkv_bias, drop_rate = drop_rate, attn_drop = attn_drop_rate, drop_path=dpr[index],
                rpe_length=img_size // patch_size, qk_scale=qk_scale, rpe=rpe, pre_norm=pre_norm, head_dim = 64
            ), depth
        )

        self.norm = nn.LayerNorm(cast(int, embed_dim)) if pre_norm else nn.Identity()
        self.head = nn.Linear(cast(int, embed_dim), num_classes) if num_classes > 0 else nn.Identity()

    @classmethod
    def get_extra_mutation_hooks(cls):
        return [MixedAbsPosEmbed.mutate, MixedClsToken.mutate]

    @classmethod
    def preset(cls, name: str):
        """Get the model space config proposed in paper."""
        name = name.lower()
        assert name in ['tiny', 'small', 'base']
        init_kwargs = {'qkv_bias': True, 'drop_rate': 0.0, 'drop_path_rate': 0.1, 'global_pool': True, 'num_classes': 1000}
        if name == 'tiny':
            init_kwargs.update({
                'search_embed_dim': (192, 216, 240),
                'search_mlp_ratio': (3.0, 3.5, 4.0),
                'search_num_heads': (3, 4),
                'search_depth': (12, 13, 14),
            })
        elif name == 'small':
            init_kwargs.update({
                'search_embed_dim': (320, 384, 448),
                'search_mlp_ratio': (3.0, 3.5, 4.0),
                'search_num_heads': (5, 6, 7),
                'search_depth': (12, 13, 14),
            })
        elif name == 'base':
            init_kwargs.update({
                'search_embed_dim': (528, 576, 624),
                'search_mlp_ratio': (3.0, 3.5, 4.0),
                'search_num_heads': (8, 9, 10),
                'search_depth': (14, 15, 16),
            })
        else:
            raise ValueError(f'Unsupported architecture with name: {name}')
        return init_kwargs

    @classmethod
    def load_strategy_checkpoint(cls, name: str, download: bool = True, progress: bool = True):
        """
        Load the related strategy checkpoints.

        Parameters
        ----------
        name : str
            Search space size, must be one of {'random-one-shot-tiny', 'random-one-shot-small', 'random-one-shot-base'}.
        download : bool
            Whether to download supernet weights. Default is ``True``.
        progress : bool
            Whether to display the download progress. Default is ``True``.

        Returns
        -------
        BaseStrategy
            The loaded strategy.
        """
        legal = ['random-one-shot-tiny', 'random-one-shot-small', 'random-one-shot-base']
        if name not in legal:
            raise ValueError(f'Unsupported name: {name}. It should be one of {legal}.')
        name = name[16:]

        # RandomOneShot is the only supported strategy for now.
        from nni.nas.strategy import RandomOneShot
        init_kwargs = cls.preset(name)
        with no_fixed_arch():
            model_sapce = cls(**init_kwargs)
        strategy = RandomOneShot(mutation_hooks=cls.get_extra_mutation_hooks())
        strategy.attach_model(model_sapce)
        weight_file = load_pretrained_weight(f"autoformer-{name}-supernet", download=download, progress=progress)
        pretrained_weights = torch.load(weight_file)
        assert strategy.model is not None
        strategy.model.load_state_dict(pretrained_weights)
        return strategy

    @classmethod
    def load_searched_model(
        cls, name: str,
        pretrained: bool = False, download: bool = True, progress: bool = True
    ) -> nn.Module:
        """
        Load the searched subnet model.

        Parameters
        ----------
        name : str
            Search space size, must be one of {'autoformer-tiny', 'autoformer-small', 'autoformer-base'}.
        pretrained : bool
            Whether initialized with pre-trained weights. Default is ``False``.
        download : bool
            Whether to download supernet weights. Default is ``True``.
        progress : bool
            Whether to display the download progress. Default is ``True``.

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

        model_factory = FixedFactory(cls, arch)
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

        if self.global_pool:
            x = torch.mean(x[:, 1:], dim=1)
        else:
            x = x[:, 0]

        x = self.head(x)

        return x
