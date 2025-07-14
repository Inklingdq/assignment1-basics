import torch
from einops import einsum

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

