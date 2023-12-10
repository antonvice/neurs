import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

class ShiftRight(nn.Module):
    """
    This module shifts the input tensor to the right by a specified number of steps.
    It's used to prevent information leakage from future tokens in the sequence.
    """
    def __init__(self, shift: int):
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor):
        if self.shift == 0:
            return x
        prefix = torch.zeros([self.shift, *x.shape[1:]], device=x.device)
        return torch.cat([prefix, x[:-self.shift]], dim=0)

class AvgPoolShortening(nn.Module):
    """
    This module performs average pooling to shorten the sequence length.
    It's a simple form of down-sampling used in the HourGlass model.
    """
    def __init__(self, k: int):
        super().__init__()
        self.pool = nn.AvgPool1d(k, stride=k)

    def forward(self, x: torch.Tensor):
        x = x.transpose(0, 2)
        x = self.pool(x)
        return x.transpose(0, 2)

class NaiveUpSampling(nn.Module):
    """
    This module performs naive up-sampling by repeating elements of the sequence.
    It's used to restore the sequence length after shortening.
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor, x_short: torch.Tensor):
        expanded = x_short.repeat_interleave(self.k, dim=0)
        return expanded[:x.shape[0], :, :]

class LinearPoolingShortening(nn.Module):
    """
    Linear Pooling Shortening reduces sequence length by linearly transforming
    concatenated embeddings. It's useful for downsampling a sequence in a more
    sophisticated way than average pooling.
    """
    def __init__(self, d_model: int, pool_size: int):
        """
        Args:
        d_model (int): The dimension of the input embeddings.
        pool_size (int): The number of adjacent embeddings to pool together.
        """
        super().__init__()
        self.pool_size = pool_size
        self.linear = nn.Linear(d_model * pool_size, d_model)

    def forward(self, x: torch.Tensor):
        # Handle padding for equal division
        if x.size(0) % self.pool_size != 0:
            padding_size = self.pool_size - (x.size(0) % self.pool_size)
            x = torch.cat([x, torch.zeros(padding_size, *x.shape[1:], device=x.device)], dim=0)

        # Concatenate, reshape, and apply linear transformation
        x = x.unfold(0, self.pool_size, self.pool_size).permute(2, 1, 0, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        x = self.linear(x)
        return x

class AttentionBasedShortening(nn.Module):
    """
    Attention-Based Shortening uses a linear pooling mechanism followed by 
    a multi-head attention layer for sequence shortening. It allows for context-aware 
    downsampling of the sequence.
    """
    def __init__(self, d_model: int, pool_size: int, n_heads: int, dropout: float):
        """
        Args:
        d_model (int): The dimension of the input embeddings.
        pool_size (int): The size for the linear pooling.
        n_heads (int): The number of heads in the multi-head attention.
        dropout (float): Dropout rate.
        """
        super().__init__()
        self.shortening = LinearPoolingShortening(d_model, pool_size)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: torch.Tensor):
        x_short = self.shortening(x)
        x_attn, _ = self.attn(x_short, x, x)
        return x_attn + self.ffn(x_short)


class LinearUpSampling(nn.Module):
    """
    Linear UpSampling increases the sequence length by projecting each embedding
    to a higher dimension and then reshaping it to extend the sequence. This approach
    provides a straightforward method for increasing sequence length.
    """
    def __init__(self, d_model: int, up_scale: int):
        """
        Args:
        d_model (int): The dimension of the input embeddings.
        up_scale (int): The factor by which to increase the sequence length.
        """
        super().__init__()
        self.up_scale = up_scale
        self.linear = nn.Linear(d_model, d_model * up_scale)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = x.view(-1, x.size(1), d_model)
        return x

class AttentionBasedUpSampling(nn.Module):
    """
    Attention-Based UpSampling combines linear upsampling with an attention mechanism.
    It allows the model to upsample the sequence in a context-aware manner, where the
    extended sequence can gather information from the shortened sequence.
    """
    def __init__(self, d_model: int, up_scale: int, n_heads: int, dropout: float):
        """
        Args:
        d_model (int): The dimension of the input embeddings.
        up_scale (int): The factor by which to increase the sequence length.
        n_heads (int): The number of heads in the multi-head attention.
        dropout (float): Dropout rate.
        """
        self.upsampling = LinearUpSampling(d_model, up_scale)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: torch.Tensor, x_short: torch.Tensor):
        x_up = self.upsampling(x_short)
        x_attn, _ = self.attn(x_up, x_short, x_short)
        return x + self.ffn(x_attn)
    
    
class HourGlass(nn.Module):
    """
    The HourGlass model is a recursive neural network structure for sequence modeling.
    It consists of layers that progressively shorten and then lengthen the sequence,
    with recursive application of the HourGlass structure in the middle layers.
    """
    def __init__(self, n_heads: int, d_model: int, dropout: float, d_ff: int, shortening_factors: list[int],
                use_linear_shortening=True, use_linear_upsampling=True):
        super().__init__()
        # Initial transformer layer
        self.pre = TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout)
        self.pre_encoder = TransformerEncoder(self.pre, num_layers=1)

        # Shortening and upsampling methods
        shortening_method = LinearPoolingShortening if use_linear_shortening else AttentionBasedShortening
        upsampling_method = LinearUpSampling if use_linear_upsampling else AttentionBasedUpSampling

        # Shortening and upsampling layers
        self.shift_right = ShiftRight(shortening_factors[0] - 1)
        self.shortening = shortening_method(d_model, shortening_factors[0], n_heads, dropout)
        self.up_sampling = upsampling_method(d_model, shortening_factors[0], n_heads, dropout)

        # Recursive HourGlass structure
        if len(shortening_factors) == 1:
            self.shortened = TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout)
            self.shortened_encoder = TransformerEncoder(self.shortened, num_layers=1)
            self.hour_glass = None
        else:
            self.hour_glass = HourGlass(n_heads, d_model, dropout, d_ff, shortening_factors[1:])

        # Final transformer layer
        self.post = TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout)
        self.post_encoder = TransformerEncoder(self.post, num_layers=1)

    def forward(self, x: torch.Tensor, src_mask=None):
        x = self.pre_encoder(x, mask=src_mask)
        x_short = self.shortening(self.shift_right(x))

        if self.hour_glass is None:
            x_short = self.shortened_encoder(x_short, mask=src_mask)
        else:
            x_short = self.hour_glass(x_short)

        x = x + self.up_sampling(x, x_short)
        x = self.post_encoder(x, mask=src_mask)
        return x

# Example usage
n_heads, d_model, dropout, d_ff, shortening_factors = 4, 512, 0.1, 2048, [2, 3]
model = HourGlass(n_heads, d_model, dropout, d_ff, shortening_factors)
x = torch.randn(10, 32, d_model)  # Example input: seq_len, batch_size, d_model
output = model(x)