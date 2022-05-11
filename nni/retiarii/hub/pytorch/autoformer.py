# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import itertools
import math
import warnings

import torch
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.", stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class RelativePosition2D(nn.Module):
    def __init__(self, head_embed_dim, length=14,) -> None:
        super().__init__()
        self.head_embed_dim = head_embed_dim
        self.legnth = length
        self.embeddings_table_v = nn.Parameter(
            torch.randn(length * 2 + 2, head_embed_dim))
        self.embeddings_table_h = nn.Parameter(
            torch.randn(length * 2 + 2, head_embed_dim))

        trunc_normal_(self.embeddings_table_v, std=.02)
        trunc_normal_(self.embeddings_table_h, std=.02)

    def forward(self, length_q, length_k):
        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        # compute the row and column distance
        distance_mat_v = (range_vec_k[None, :] //
                          int(length_q ** 0.5) -
                          range_vec_q[:, None] //
                          int(length_q ** 0.5))
        distance_mat_h = (range_vec_k[None, :] %
                          int(length_q ** 0.5) -
                          range_vec_q[:, None] %
                          int(length_q ** 0.5))
        # clip the distance to the range of [-legnth, legnth]
        distance_mat_clipped_v = torch.clamp(
            distance_mat_v, -self.legnth, self.legnth)
        distance_mat_clipped_h = torch.clamp(
            distance_mat_h, -self.legnth, self.legnth)

        # translate the distance from [1, 2 * legnth + 1], 0 is for the cls
        # token
        final_mat_v = distance_mat_clipped_v + self.legnth + 1
        final_mat_h = distance_mat_clipped_h + self.legnth + 1
        # pad the 0 which represent the cls token
        final_mat_v = torch.nn.functional.pad(
            final_mat_v, (1, 0, 1, 0), "constant", 0)
        final_mat_h = torch.nn.functional.pad(
            final_mat_h, (1, 0, 1, 0), "constant", 0)

        final_mat_v = torch.tensor(
            final_mat_v,
            dtype=torch.long,
            device=self.embeddings_table_v.device)
        final_mat_h = torch.tensor(
            final_mat_h,
            dtype=torch.long,
            device=self.embeddings_table_v.device)
        # get the embeddings with the corresponding distance
        embeddings = self.embeddings_table_v[final_mat_v] + \
            self.embeddings_table_h[final_mat_h]

        return embeddings


class Attention(nn.Module):
    def __init__(
            self,
            embed_dim,
            fixed_embed_dim,
            num_heads,
            attn_drop=0.,
            proj_drop=0,
            rpe=False,
            qkv_bias=False,
            qk_scale=None,
            rpe_length=14) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpe = rpe
        if rpe:
            self.rel_pos_embed_k = RelativePosition2D(
                fixed_embed_dim // num_heads, rpe_length)
            self.rel_pos_embed_v = RelativePosition2D(
                fixed_embed_dim // num_heads, rpe_length)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B,
            N,
            3,
            self.num_heads,
            C //
            self.num_heads).permute(
            2,
            0,
            3,
            1,
            4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.rpe:
            r_p_k = self.rel_pos_embed_k(N, N)
            attn = attn + (
                q.permute(
                    2, 0, 1, 3).reshape(
                    N, self.num_heads * B, -1) @ r_p_k.transpose(
                    2, 1)) .transpose(
                1, 0).reshape(
                        B, self.num_heads, N, N) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.rpe:
            r_p_v = self.rel_pos_embed_v(N, N)
            attn_1 = attn.permute(
                2, 0, 1, 3).reshape(
                N, B * self.num_heads, -1)
            # The size of attention is (B, num_heads, N, N), reshape it to (N, B*num_heads, N) and do batch matmul with
            # the relative position embedding of V (N, N, head_dim) get shape like (N, B*num_heads, head_dim). We reshape it to the
            # same size as x (B, num_heads, N, hidden_dim)
            x = x + (attn_1 @ r_p_v).transpose(1, 0).reshape(B,
                                                             self.num_heads, N, -1).transpose(2, 1).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        fixed_embed_dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        rpe=False,
        drop_rate=0.,
        attn_drop=0.,
        proj_drop=0.,
        drop_path=0.,
        pre_norm=True,
        rpe_length=14,
    ):
        super().__init__()

        self.normalize_before = pre_norm
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.dropout = drop_rate
        self.attn = Attention(
            embed_dim=embed_dim,
            fixed_embed_dim=fixed_embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            rpe=rpe,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            rpe_length=rpe_length)

        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(
            embed_dim, nn.ValueChoice.to_int(
                embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(
            nn.ValueChoice.to_int(
                embed_dim * mlp_ratio),
            embed_dim)

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
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


@model_wrapper
class AutoformerSpace(nn.Module):
    """
    The search space that is proposed in Autoformer.
    There are four searchable variables: depth, embedding dimension, heads number and MLP ratio.

    Parameters
    ----------
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
    search_embed_dim : list of int
        The search space of embedding dimension.
    search_mlp_ratio : list of float
        The search space of MLP ratio.
    search_num_heads : list of int
        The search space of number of heads.
    search_depth: list of int
        The search space of depth.

    References
    ----------
    .. Chen M, Peng H, Fu J, et al. Autoformer: Searching transformers for visual recognition. CVPR 2021: 12270-12280.
    """

    def __init__(
            self,
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
            qk_scale: float = None,
            rpe: bool = True,
            search_embed_dim: list = [
                192,
                216,
                240],
            search_mlp_ratio: list = [
                3.5,
                4.0],
            search_num_heads: list = [
                3,
                4],
            search_depth: list = [
                12,
                13,
                14],
    ):
        super().__init__()

        embed_dim = nn.ValueChoice(search_embed_dim, label="embed_dim")
        fixed_embed_dim = nn.ModelParameterChoice(
            search_embed_dim, label="embed_dim")
        depth = nn.ValueChoice(search_depth, label="depth")
        self.patch_embed = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size)
        self.patches_num = int((img_size // patch_size) ** 2)
        self.global_pool = global_pool
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fixed_embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        self.blocks = []
        dpr = [
            x.item() for x in torch.linspace(
                0,
                drop_path_rate,
                max(search_depth))]  # stochastic depth decay rule

        self.abs_pos = abs_pos
        if self.abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(
                1, self.patches_num + 1, fixed_embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        self.blocks = nn.Repeat(lambda index: nn.LayerChoice([
            TransformerEncoderLayer(embed_dim=embed_dim,
                                    fixed_embed_dim=fixed_embed_dim,
                                    num_heads=num_heads, mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, drop_rate=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[index],
                                    rpe_length=img_size // patch_size,
                                    qk_scale=qk_scale, rpe=rpe,
                                    pre_norm=pre_norm,)
            for mlp_ratio, num_heads in itertools.product(search_mlp_ratio, search_num_heads)
        ], label=f'layer{index}'), depth)

        self.pre_norm = pre_norm
        if self.pre_norm:
            self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(
            embed_dim,
            num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1).view(B, self.patches_num, -1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.abs_pos:
            x = x + self.pos_embed
        x = self.blocks(x)
        if self.pre_norm:
            x = self.norm(x)
        if self.global_pool:
            x = torch.mean(x[:, 1:], dim=1)
        else:
            x = x[:, 0]

        x = self.head(x)

        return x
