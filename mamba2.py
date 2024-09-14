"""Simple, minimal implementation of Mamba2 in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4
    [3] Mamba2: Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Tri Dao, Albert Gu)
        https://arxiv.org/abs/2405.21060
    [4] Mamba2, Blog:
        https://tridao.me/blog/2024/mamba2-part1-model/
        https://tridao.me/blog/2024/mamba2-part2-theory/
        https://tridao.me/blog/2024/mamba2-part3-algorithm/
        https://tridao.me/blog/2024/mamba2-part4-systems/

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size

"""
from __future__ import annotations
import math
import json
import torch # For loading pretrained weights
import jax
import jax.numpy as np
from jax.nn.initializers import lecun_normal, normal 
import flax
from flax import linen as nn
from flax.linen.initializers import normal as flax_normal
from dataclasses import dataclass
from einops import rearrange, repeat, einsum

from typing import Union, NamedTuple, TypeAlias

import math

from pytorch_to_jax import convert_from_pytorch

from torch import LongTensor, Tensor

Device: TypeAlias = str | torch.device | None


@dataclass
class ModelArgs_Mamba2: # The same as torch version since this does not have any torch specific code
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    # dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    # conv_bias: bool = True
    # bias: bool = False
    headdim: int = 64
    chunk_size: int = 64
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        
        # if self.dt_rank == 'auto':
        #     self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class InferenceCache(NamedTuple):
    conv_state: np.ndarray  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: np.ndarray  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: ModelArgs_Mamba2, device=None):
        """
        Allocate memory for conv_state and ssm_state, using JAX.
        In JAX, there is no direct 'device' handling like in PyTorch, since it uses
        backend mechanisms internally for device placement.
        """
        return InferenceCache(
            np.zeros(
                (batch_size, args.d_inner + 2 * args.d_state, args.d_conv),
                dtype=np.float32
            ),
            np.zeros(
                (batch_size, args.nheads, args.headdim, args.d_state),
                dtype=np.float32
            ),
        )

# class InferenceCache(NamedTuple):
#     conv_state: Tensor  # (batch, d_inner + 2 * d_state, d_conv)
#     ssm_state: Tensor  # (batch, nheads, headdim, d_state)

#     @staticmethod
#     def alloc(batch_size: int, args: ModelArgs_Mamba2, device: Device = None):
#         return InferenceCache(
#             torch.zeros(
#                 batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
#             ),
#             torch.zeros(
#                 batch_size, args.nheads, args.headdim, args.d_state, device=device
#             ),
#         )


class Mamba2(nn.Module):
    args: ModelArgs_Mamba2

    def setup(self):
        """Full Mamba model."""
        super().__init__()
    
        self.embedding = nn.Embed(self.args.vocab_size, self.args.d_model)
        self.layers = [ResidualBlock(self.args) for _ in range(self.args.n_layer)]
        self.norm_f = RMSNorm(self.args.d_model)

    def attend(self, input):
        """Use for weight sharing to produce output logits of model"""
        return self.embedding.attend(input)

    @nn.compact
    def __call__(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = self.attend(x)

        return logits


    @staticmethod
    def from_pretrained(pretrained_model_name: str, tokenizer=None):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba2-2.7b'
                * 'state-spaces/mamba2-1.3b'
                * 'state-spaces/mamba2-780m'
                * 'state-spaces/mamba2-370m'
                * 'state-spaces/mamba2-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """

        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            assert resolved_archive_file, "Failed to get huggingface config file"
            return json.load(open(resolved_archive_file))
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            assert resolved_archive_file, "Failed to get huggingface state dict file"
            return torch.load(resolved_archive_file, weights_only=True, map_location=torch.device('cpu'), mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs_Mamba2(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size'],
            pad_vocab_size_multiple=config_data["pad_vocab_size_multiple"],
        )   
        # TODO
        print("args is", args)

        model = Mamba2(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', 'params.')
            new_state_dict[new_key] = state_dict[key]
        
        rng = jax.random.PRNGKey(7)
        input_ids = tokenizer("hello how are you" * 256, return_tensors='pt').input_ids
        input_ids = np.array(input_ids.numpy())
        random_params = model.init(rng, input_ids)
        random_params_flatten = flax.traverse_util.flatten_dict(random_params, sep=".")

        params = convert_from_pytorch(new_state_dict, random_params_flatten)
        
        return model, params


class ResidualBlock(nn.Module):
    args:ModelArgs_Mamba2
    # include other necessary parameters from ModelArgs_Mamba2 if needed

    def setup(self):
        """Full Mamba model."""
        super().__init__()
        self.mixer = Mamba2Block(self.args)
        self.norm = RMSNorm(self.args.d_model)

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output = self.mixer(self.norm(x)) + x
        return output


class Mamba2Block(nn.Module):
    args: ModelArgs_Mamba2

    def setup(self):
        d_in_proj = 2 * self.args.d_inner + 2 * self.args.d_state + self.args.nheads
        self.in_proj = nn.Dense(features=d_in_proj, 
                                kernel_init=normal(), 
                                use_bias=False,
                                )
        
        conv_dim = self.args.d_inner + 2 * self.args.d_state
        print("conv_dim is", conv_dim)
        self.conv1d = nn.Conv(
            features=conv_dim,
            kernel_size=[self.args.d_conv],
            feature_group_count=conv_dim,
            padding=self.args.d_conv-1,
            use_bias=True,
            )

        self.dt_bias = nn.Dense(features=self.args.nheads, use_bias=True)
        self.A_log = nn.Dense(features=self.args.nheads, use_bias=True)
        self.D = nn.Dense(features=self.args.nheads, use_bias=True)
        self.norm = RMSNorm(self.args.d_inner)
        self.out_proj = nn.Dense(self.args.d_model, kernel_init=normal(), use_bias=False)


    def __call__(self, x, h: InferenceCache | None = None):
        """Mamba2 block forward.
    
        Args:
            x: inputs of shape (b, l, d_model)
            h: shape (b, d_model), initialized to 0s if not present. 
    
        Returns:
            output: shape (b, l, d_model)
            h: updated state after processing inputs
        """
        if h is not None:
            return self.step(x, h)
        else:
            A = -np.exp(self.A_log(x)) # (nheads, )
            zxbcdt = self.in_proj(x) # (b, l, d_in_proj)
            # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.split.html
            z, xBC, dt = np.split(zxbcdt, [self.args.d_inner, 2 * self.args.d_inner + 2 * self.args.d_state], axis=-1)
        dt = jax.nn.softplus(dt + self.dt_bias(x))

        # Pad or truncate xBC seqlen to d_conv
        conv_state = np.pad(rearrange(xBC, "b l d -> b d l"), max(self.args.d_conv - x.shape[1], 0)) # Incorrect for now

        xBC = jax.nn.silu(
            self.conv1d(xBC)[:, : x.shape[1], :]
        )
        x, B, C = np.split(
            xBC, [self.args.d_inner, self.args.d_inner + self.args.d_state], axis=-1
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        y, ssm_state = ssd(
            x * np.expand_dims(dt, -1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
        )

        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        h = InferenceCache(conv_state, ssm_state)

        return y, h


    def step(self, x):
        pass


def segsum(x):
    """More stable segment sum calculation."""
    # T = x.size(-1)
    T = x.shape[-1]
    x = repeat(x, "... d -> ... d e", e=T)
    # mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    mask = np.tril(np.ones((T, T), dtype=bool), k=-1)  
    # x = x.masked_fill(~mask, 0)
    x = np.where(mask, x, 0)
    # x_segsum = torch.cumsum(x, dim=-2)
    x_segsum = np.cumsum(x, axis=-2)
    # mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    mask = np.tril(np.ones((T, T), dtype=bool), k=0)
    # x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    x_segsum = np.where(mask, x_segsum, -np.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None):

    """Structed State Space Duality (SSD) - the core of Mamba-2

    This is almost the exact same minimal SSD code from the blog post.

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)

    Source
    1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
    2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
    """
    print(x.shape, chunk_size)
    assert x.shape[1] % chunk_size == 0

    # Rearrange into chunks
    x, A, B, C = [rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    # A_cumsum = torch.cumsum(A, dim=-1)
    A_cumsum = np.cumsum(A, axis=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    # L = torch.exp(segsum(A, device=device))
    L = np.exp(segsum(A))
    # Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)
    Y_diag = np.einsum("b c l h n, b c s h n, b h c l s, b c s h p -> b c l h p", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    # decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    decay_states = np.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    # states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)
    state = np.einsum("b c l h n, b h c l, b c l h p -> b c h p n", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        # initial_states = torch.zeros_like(states[:, :1])
        initial_states = np.zeros_like(state[:, :1])
    # states = torch.cat([initial_states, states], dim=1)
    states = np.concatenate([initial_states, states], axis=1)
    # decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    decay_chunk = np.exp(segsum(np.pad(A_cumsum[:, :, :, -1], ((0, 0), (1, 0)))))
    # new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    new_states = np.einsum("b h z c, b c h p n -> b z h p n", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    # state_decay_out = torch.exp(A_cumsum)
    state_decay_out = np.exp(A_cumsum)
    # Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)
    Y_off = np.einsum("b c l h n, b c h p n, b h c l -> b c l h p", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state
    

class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x, z=None):
        if z is not None:
            x = x * jax.nn.silu(z)

        weight = self.param('weight', nn.initializers.ones, (self.d_model,)) 
        normed = x * jax.lax.rsqrt(np.mean(np.square(x), axis=-1, keepdims=True) + self.eps)
        output = normed * weight
        return output


if __name__ == '__main__':
    # Test for RMSNorm
    
    # Generate a random example input
    rng = jax.random.PRNGKey(0)
    input_shape = (10, 20)  # example shape
    x = jax.random.normal(rng, input_shape)

    # Initialize the model
    d_model = 20  # should match the last dimension of the input
    rms_norm = RMSNorm(d_model=d_model)

    # Initialize parameters
    params = rms_norm.init(rng, x)

    # Apply the model
    output = rms_norm.apply(params, x)

    print("Input:", x)
    print("Output:", output)

