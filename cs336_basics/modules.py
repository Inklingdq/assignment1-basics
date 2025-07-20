import math
from jaxtyping import Float
import torch
from einops import einsum, rearrange, reduce
from torch import Tensor

class Linear(torch.nn.Module):
    """
    A simple linear layer that applies a linear transformation to the input.
    """
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        torch.nn.init.normal_(self.W, 0, 2/(in_features + out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear transformation to the input tensor `x`.
        """
        return einsum(x, self.W, "... i, j i->... j")
    
class Embedding(torch.nn.Module):
    """
    A simple embedding layer that maps input indices to dense vectors.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.normal_(self.weight, 0, 2/(embedding_dim + num_embeddings))

    def forward(self, indices: torch.LongTensor) -> torch.Tensor:
        """
        Maps input indices to their corresponding embeddings.
        """
        return self.weight[indices]

class RMSNorm(torch.nn.Module):
    """
    A simple RMSNorm layer that normalizes the input tensor.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS normalization to the input tensor `x`.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        xsquare = einsum(x, x, "... i, ... i -> ... i")
        mean_square = reduce(xsquare, "... i -> ...", "mean")
        denominator = rearrange(torch.sqrt(mean_square + self.eps), "... -> ... 1")
        result = einsum(x, self.g,"... i, i -> ... i") / denominator
        return result.to(in_dtype)
    
class SwiGLU(torch.nn.Module):
    """
    A simple SwiGLU feed-forward network.
    """
    def __init__(self, d_model: int, d_ff:int, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = torch.nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2 = torch.nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.w3 = torch.nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype)) 
        self.device = device
        self.dtype = dtype
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the SwiGLU transformation to the input tensor `x`.
        """
        x1 = einsum(x, self.w1, "... i, j i -> ... j")
        x1 = einsum(torch.sigmoid(x1), x1, "..., ... -> ...")
        x3 = einsum(x, self.w3, "... i, j i -> ... j")
        x2 = einsum(x1, x3, "... i, ... i -> ... i")
        return einsum(x2, self.w2, "... j, i j -> ... i")

class RotaryPositionalEmbedding(torch.nn.Module):
    """
    A simple Rotary Positional Embedding layer.
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        inv_freq = 1 / (theta ** (2 * torch.arange(0, d_k // 2, device=device) / d_k))
        positions = torch.arange(max_seq_len, device=device)
        angle = einsum(positions, inv_freq, "i, j -> i j")

        self.cos = angle.cos()
        self.sin = angle.sin()

        self.register_buffer("cos_rot", self.cos, persistent=False)
        self.register_buffer("sin_rot", self.sin, persistent=False)
    def apply_rotary(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        """
        Applies the rotary positional embedding to the input tensor `x` using the provided sine and cosine
        """
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x_rot = torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim = -1)
        return x_rot.flatten(-2)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Applies Rotary Positional Embedding to the input tensor `x` at the specified token positions.
        """
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]
        while sin.dim() < x.dim() - 1:
            sin = sin.unsqueeze(0)
            cos = cos.unsqueeze(0)
        return self.apply_rotary(x, sin, cos)

def softmax(x: torch.Tensor, i: int):
    '''
    Apply softmax operationon the ith dimension of the input tensor.
    '''
    x_max, _ = torch.max(x, dim = i, keepdim = True)
    x_exp = torch.exp(x - x_max)
    return x_exp/x_exp.sum(dim = i, keepdim=True)

def scaled_dot_product_attetion(Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None) -> Float[Tensor, " ... queries d_v"]:
    '''
    Implement scaled dot product attention.
    '''
    dk = K.shape[-1]
    scores = einsum(Q, K, "...  q d_k, ... k d_k -> ... q k")/math.sqrt(dk)
    if mask is not None:
        if mask.shape[-2:] != scores.shape[-2:]:
            mask = rearrange(mask, "... i j -> ... j i")
        scores = scores.masked_fill(mask == 0, -math.inf) 
    attention_probability = softmax(scores, -1)
    return einsum(attention_probability, V, "... q k, ... k dv -> ... q dv")
