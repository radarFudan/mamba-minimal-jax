"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

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
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
import torch # For loading pretrained weights
import jax
import jax.numpy as np
from jax.nn.initializers import lecun_normal, normal # TODO, check the difference between lecun_normal, flax_normal, normal. 
import flax
from flax import linen as nn
from flax.linen.initializers import normal as flax_normal
from dataclasses import dataclass
from einops import rearrange, repeat, einsum

from typing import Union

import math

from pytorch_to_jax import convert_from_pytorch


@dataclass
class ModelArgs: # The same as torch version since this does not have any torch specific code
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):
    args: ModelArgs

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
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location=torch.device('cpu'), mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )   

        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', 'params.')
            new_state_dict[new_key] = state_dict[key]
        
        rng = jax.random.PRNGKey(7)
        input_ids = tokenizer("hello what is your name", return_tensors='pt').input_ids
        input_ids = np.array(input_ids.numpy())
        random_params = model.init(rng, input_ids)
        random_params_flatten = flax.traverse_util.flatten_dict(random_params, sep=".")

        params = convert_from_pytorch(new_state_dict, random_params_flatten)
        
        return model, params


class ResidualBlock(nn.Module):
    args:ModelArgs
    # include other necessary parameters from ModelArgs if needed

    def setup(self):
        """Full Mamba model."""
        super().__init__()
        self.mixer = MambaBlock(self.args)
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


class MambaBlock(nn.Module):
    args: ModelArgs

    def setup(self):
        self.in_proj = nn.Dense(features=self.args.d_inner * 2, 
                                kernel_init=normal(), 
                                use_bias=self.args.bias)
        
        # Adjusted for Flax. Flax does not have nn.Conv1d, so you might need to reshape or use a different approach
        self.conv1d = nn.Conv(features=self.args.d_inner,
                              kernel_size=[self.args.d_conv],
                              feature_group_count=self.args.d_inner,
                              padding=self.args.d_conv - 1,
                              use_bias=self.args.conv_bias,
                              )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Dense(self.args.dt_rank + self.args.d_state * 2, use_bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Dense(self.args.d_inner, use_bias=True)

        A = np.tile(np.arange(1, self.args.d_state + 1), (self.args.d_inner, 1))
        self.A_log = self.param('A_log', lambda rng, shape: np.log(A), (self.args.d_inner, self.args.d_state))
        self.D = self.param('D', nn.initializers.ones, (self.args.d_inner,))
        self.out_proj = nn.Dense(self.args.d_model, kernel_init=normal(), use_bias=self.args.bias)


    def __call__(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)

        # The split_size is converted to the indices_or_sections method, notice the difference! 
        (x, res) = np.split(x_and_res, indices_or_sections=[self.args.d_inner,], axis=-1)

        # TODO, summarize the difference between torch and jax convolution! 
        x = self.conv1d(x)[:, :l, :]

        x = jax.nn.silu(x)

        y = self.ssm(x)
        
        y = y * jax.nn.silu(res)
        
        output = self.out_proj(y)

        return output


    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        # TODO, There is a type conversion to float in the torch version s
        A = -np.exp(self.A_log)  # shape (d_in, n)
        D = self.D

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        # The split_size is converted to the indices_or_sections method, notice the difference! 
        (delta, B, C) = np.split(x_dbl, indices_or_sections=[self.args.dt_rank, self.args.dt_rank+n], axis=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = jax.nn.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y


    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        deltaA = np.exp(np.einsum('b l d, d n -> b l d n', delta, A))
        deltaB_u = np.einsum('b l d, b l n, b l d -> b l d n', delta, B, u)

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        x = np.zeros((b, d_in, n))
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = np.einsum('b d n, b n -> b d', x, C[:, i, :])
            ys.append(y)
        y = np.stack(ys, axis=1)  # shape (b, l, d_in)
        
        y = y + u * D

        return y


class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', nn.initializers.ones, (self.d_model,)) # TODO, maybe use setup will be more clear
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

