import torch
import torch.nn as nn
import torch.nn.init as init
from torch.functional import F


class AttentionConv(nn.Module):
    """ Attention Convolution as defined in the paper https://arxiv.org/abs/1906.05909. The definition in this class works
    for every case, but is quite inefficient with high numbers of heads and groups. When heads==groups use AttentionConvFull.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, heads=8, device=None):
        """ 
        Parameters
        ----------
        in_channels : int
            number of input channels, i.e. the size of single input embedding vectors. Must be divisible by <groups>.
        out_channels : int
            number of output channels, i.e. the size of single output embedding vectors. Must be divisible by <heads>.
        kernel_size : int
            width and height of the attention window. Default 3.
        stride : int
            not implemented. Default 1.
        groups : int
            number of groups for the QKV linear projections: input vectors are separated into <groups> parts
            of size <in_channels>/<groups> and each part goes through a separate linear projection
            (different groups are not connected). Default 1.
        heads : int
            number of heads when computing attention: after projecting onto QKV, vectors are separated into
            <heads> parts of size <out_channels>/<heads> and each part is used to compute a separate attention
            score for a separate part of VALUE vectors. Default 8.
        device : None
            computing device to be used.
        """
        super(AttentionConv, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # add padding as half the kernel size
        self.padding = kernel_size // 2
        # groups regulate the subdivision for the initial projection
        self.groups = groups
        # heads are the attention heads when computing multihead attention
        self.heads = heads
        # others
        self.device = device

        def assert_divisible(a, b, stra, strb):
          assert in_channels % self.groups == 0, f'{stra} should be divisible by {strb} (example: {stra}: 40, {strb}: 4). Found {stra}={a}, {strb}={b}.'

        assert_divisible(in_channels, groups, 'in_channels', 'groups')
        assert_divisible(out_channels, heads, 'out_channels', 'heads')

        # relative embedding base on distances in receptive fields
        # B|H|W|K|K|h|c
        # 1,1,1,k,k,h,c
        self.rel_emb = nn.Parameter(torch.randn(1, 1, 1, kernel_size, kernel_size, heads, out_channels // heads), requires_grad=True)
        # B|H|W|h|c
        # 1,1,1,h,c
        self.q_emb = nn.Parameter(torch.randn(1, 1, 1, heads, out_channels // heads), requires_grad=True)

        # projections of kqv
        # convolutions with kernel_size=1 is the same as a fully connected layer
        # introducing groups separate the "pixel" features into self attention heads
        self.key_conv =   nn.Conv2d (in_channels, out_channels, padding=self.padding,   kernel_size=1, groups=groups, bias=False)
        self.query_conv = nn.Conv2d (in_channels, out_channels, padding=0,              kernel_size=1, groups=groups, bias=False)
        self.value_conv = nn.Conv2d (in_channels, out_channels, padding=self.padding,   kernel_size=1, groups=groups, bias=False)

        self.attention_scores = nn.Softmax(dim=-1)

        self.reset_parameters()

    def forward(self, x):
        """ Compute convolutional/local self-attention.
            
        Parameters
        ----------
        x : Tensor
            channel-last input tensor of shape (B,H,W,C)

        Returns
        -------
        out : Tensor
            channel-last output tensor of shape (B,H,W,C) (same shape as the input)
        """

        # (B,H,W,C)
        batch, height, width, channels = x.size()
        h_channels = self.out_channels // self.heads

        def compute_conv(grid, conv_layer):
          # permute to channel-first -> project with convolution -> will be shape (B,H+2p,W+2p,C), having already concatenated g groups of size C//g ->
          # -> permute to channel-last -> separate channels into heads and sub-channels -> will be of shape (B,H+2p,W+2p,h,c)
          grid_out = conv_layer(grid.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
          return grid_out.reshape(*grid_out.size()[:-1], self.heads, h_channels).contiguous()

        # compute projections
        q_out = compute_conv(x, self.query_conv)    # (B,H,W,h,c)
        k_out = compute_conv(x, self.key_conv)      # (B,H+2p,W+2p,h,c)
        v_out = compute_conv(x, self.value_conv)    # (B,H+2p,W+2p,h,c)

        # prepare for matmul
        q_out = q_out.unsqueeze(-2)           # (B,H,   W,   h,1,c)
        k_out = k_out.unsqueeze(-1)           # (B,H+2p,W+2p,h,c,1)
        rel_emb = self.rel_emb.unsqueeze(-1)  # (B,H,  ,W   ,k,k,h,c,1)

        # compute attn weights dot-products
        attn_weights = []
        size_i, size_j = height + 2*self.padding - self.kernel_size, width + 2*self.padding - self.kernel_size
        for i in range(self.kernel_size):
          for j in range(self.kernel_size):
            # matmul (B,H,W,h,1,c) @ (B,H,W,h,c,1)
            term_i, term_j = size_i+i+1, size_j+j+1
            elem = q_out @ (k_out[:, i:term_i, j:term_j] + rel_emb[:,:,:,i,j]) # (B,H,W,h,1,1)
            elem = elem.squeeze(-1).squeeze(-1) # (B,H,W,h)
            attn_weights.append(elem)

        # stack attn weights, compute softmax, arrange in a kernel-like grid
        attn_weights = torch.stack(attn_weights, dim=-1)   # (B,H,W,h,k*k)
        attn_weights = self.attention_scores(attn_weights) # (B,H,W,h,k*k)
        attn_weights = attn_weights.view(batch, height, width, self.heads, 1, self.kernel_size, self.kernel_size) # (B,H,W,h,1,k,k)

        # compute weighted sum
        out = torch.zeros(batch, height, width, self.heads, h_channels, device=self.device)
        size_i, size_j = height+2*self.padding-self.kernel_size, width+2*self.padding-self.kernel_size
        for i in range(self.kernel_size):
          for j in range(self.kernel_size):
            term_i, term_j = size_i+i+1, size_j+j+1
            out += (attn_weights[:, :, :, :, :, i, j] * v_out[:, i:term_i, j:term_j])

        out = out.reshape(batch, height, width, channels) # B,H,W,C

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_emb, 0, 1)
        init.normal_(self.q_emb, 0, 1)



class AttentionConvFull(nn.Module):
    """ Attention Convolution as defined in the paper https://arxiv.org/abs/1906.05909. The definition correspond to the case:
    heads==out_channels, but is much more time-efficient than AttentionConv, with the drawback of being less space-efficient.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False, device=None):
        """ 
        Parameters
        ----------
        in_channels : int
            number of input channels, i.e. the size of single input embedding vectors. Must be divisible by <groups>.
        out_channels : int
            number of output channels, i.e. the size of single output embedding vectors. Must be divisible by <heads>.
        kernel_size : int
            width and height of the attention window. Default 3.
        stride : int
            not implemented. Default 1.
        groups : int
            number of groups for the QKV linear projections: input vectors are separated into <groups> parts
            of size <in_channels>/<groups> and each part goes through a separate linear projection
            (different groups are not connected). Default 1.
        device : None
            computing device to be used.
        """
        super(AttentionConvFull, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # add padding as half the kernel size
        self.padding = kernel_size // 2
        self.groups = groups

        # others
        self.device = device

        assert in_channels % self.groups == 0, "in_channels should be divided by groups. (example: in_channels: 40, groups: 4)"
        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        # relative embedding base on distances in receptive fields
        self.rel_emb = nn.Parameter(torch.randn(groups, out_channels // groups, 1, kernel_size, kernel_size), requires_grad=True)
        self.q_emb = nn.Parameter(torch.randn(out_channels, 1, 1), requires_grad=True)

        # projections of kqv
        # convolutions with kernel_size=1 is the same as a fully connected layer
        # introducing groups separate the "pixel" features into self attention heads
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)

        self.attention_scores = nn.Softmax(dim=-1)

        self.reset_parameters()

    def forward(self, x):
        """ Compute convolutional/local self-attention.
            
        Parameters
        ----------
        x : Tensor
            channel-last input tensor of shape (B,H,W,C)

        Returns
        -------
        out : Tensor
            channel-last output tensor of shape (B,H,W,C) (same shape as the input)
        """

        # (B, C, H, W)
        x = x.permute((0, 3, 1, 2)) # channel first

        batch, channels, height, width = x.size()

        # apply padding
        # (B, C, H+2p, W+2p)
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        # compute projections
        q_out = self.query_conv(x)        # (B, C, H   , W)
        k_out = self.key_conv(padded_x)   # (B, C, H+2p, W+2p)
        v_out = self.value_conv(padded_x) # (B, C, H+2p, W+2p)

        def to_kernels_each_head(t):
            # (B, C, H, W, k, k)
            t = t.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
            # (B, h, c, H*W, k, k)
            return t.reshape((t.shape[0], self.groups, self.out_channels // self.groups, t.shape[-4] * t.shape[-3], t.shape[-2], t.shape[-1]))

        # unfold keys and values into kernels and divide channels for each head, also flatten along height and width axis when indexing kernels
        k_out = to_kernels_each_head(k_out) # (B, h, c, H*W, k, k)
        v_out = to_kernels_each_head(v_out) # (B, h, c, H*W, k, k)

        # sum keys with relative position embeddings
        k_out = k_out + self.rel_emb  # (B, h, c, H*W, k, k)
        q_out = q_out + self.q_emb    # (B, C, H, W)

        # flatten along kernel dimensions
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1) # (B, h, c, H, W, k*k)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1) # (B, h, c, H, W, k*k)
        # shape queries in order to multiply with keys along kernels dimension
        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)  # (B, h, c, H, W, 1)

        # multiply
        out = q_out * k_out # (B, h, c, H, W, 1) * # (B, h, c, H, W, k*k) -> (B, h, c, H, W, k*k)
        # apply softmax
        # (B, h, c, H, W, k*k)
        out_weights = self.attention_scores(out)
        # sum over weighted values
        # (B, h, c, H, W, k*k) -> (B, h, c, H, W) -> (B, C, H, W)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out_weights, v_out).view(batch, -1, height, width)

        # (B, H, W, C)
        out = out.permute((0, 2, 3, 1)) # channel last

        return out


    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_emb, 0, 1)
        init.normal_(self.q_emb, 0, 1)