import math
from typing import Optional
from jaxtyping import Float
import torch
from einops import einsum, rearrange, reduce
from torch import Tensor
from torch import nn

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
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff) 
        self.device = device
        self.dtype = dtype
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the SwiGLU transformation to the input tensor `x`.
        """
        x1 = self.w1(x)
        x1 = einsum(torch.sigmoid(x1), x1, "..., ... -> ...")
        x3 = self.w3(x)
        x2 = einsum(x1, x3, "... i, ... i -> ... i")
        return self.w2(x2)

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

def scaled_dot_product_attention(Q: Float[Tensor, " ... queries d_k"],
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


    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int | None = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_weight = Linear(d_model, 3 * d_model)
        self.o_proj_weight = Linear(d_model, d_model)

        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len)
        else:
            self.rope = None

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        token_positions: Tensor | None = None
    ) -> Float[Tensor, "batch seq d_model"]:
        B, S, _ = x.shape

        qkv = self.qkv_weight(x)
        qkv = rearrange(qkv, "B S (three h d) -> three B h S d", three = 3, h = self.num_heads, d = self.d_model//self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if available
        if self.rope and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.tril(torch.ones(S, S), diagonal = 0)

        # Scaled dot-product attention
        attn = scaled_dot_product_attention(q, k, v, mask)

        # Merge heads
        attn = rearrange(attn, "b h s d -> b s (h d)")

        # Output projection
        return self.o_proj_weight(attn)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
       super().__init__()
       self.norm1 = RMSNorm(d_model)
       self.norm2 = RMSNorm(d_model)
       self.attention = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
       self.feedforward = SwiGLU(d_model, d_ff)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        if token_positions is None:
            seq_len = x.size(1)  # assuming x shape is (batch, seq, d_model)
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.attention(self.norm1(x), token_positions)
        x = x + self.feedforward(self.norm2(x))
        return x
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, theta: float):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.num_layers = num_layers
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, theta, context_length) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        for i in range(self.num_layers):
            x = self.transformer_blocks[i](x)
        x = self.norm(x)
        return self.lm_head(x)

def cross_entropy(logits: torch.Tensor, target: torch.Tensor):
    logits = logits - logits.max(dim=-1, keepdim=True).values
    loss = -logits[torch.arange(logits.size(0)), target].mean()
    logits = torch.exp(logits)
    logits = einsum(logits, "batch vocab_size -> batch")
    logits = torch.log(logits)
    loss += logits.mean()
    return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float, betas = (0.9, 0.99), eps = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        defaults = {"lr": lr, "weight_decay": weight_decay, "beta1": beta1, "beta2": beta2, "eps": eps}
        super().__init__(params, defaults)
    def step(self, closure: Optional[callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group['beta1'], group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t = state.get("t", 1)
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                lrt = lr * math.sqrt(1 - beta2**t)/(1- beta1**t)
                p.data = p.data - lrt * m/(torch.sqrt(v) + eps)
                p.data = p.data - lr * weight_decay * p.data
                state['m'] = m
                state['v'] = v
                state['t'] = t + 1
        return loss
