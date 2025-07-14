import torch
from einops import einsum, rearrange, reduce

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
