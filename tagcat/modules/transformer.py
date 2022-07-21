import torch.nn as nn
from .attention import AttentionConv, AttentionConvFull

################################################################################################################
############################################### UTILITY FUNCTIONS ##############################################
################################################################################################################

def extract_top_list(grid):
    return grid[:, 0, :, :]

################################################################################################################
############################################# APPLY BLOCK FUNCTION #############################################
################################################################################################################

LAYER_NORM_PRE = 'pre'
LAYER_NORM_POST = 'post'
LAYER_NORM_NONE = 'none'
LAYER_NORM_POSITION_MODES = [
    LAYER_NORM_PRE,
    LAYER_NORM_POST,
    LAYER_NORM_NONE
]


def apply_block(src, block, params, in_function=None, in_args={}, out_function=None, out_args={}):
    """ Apply the following pipeline to src:
        1 - normalize if the normalization position is 'pre' (pre-norm)
        2 - input gating. Options:
            2a - reset gate to selectively forget parts of src
            2b - everything flows through
        3 - apply <in_function> with <in_args> to preprocess src before the <block>
        4 - finally apply the <block> to src
        5 - apply <out_function> with <out_args> to preprocess src after the <block>
        6 - output gating. Options:
            6a - highway to selectively update the input
            6b - residuals to sum the input with the output as usual
        7 - normalize if the normalization position is 'post' (post-norm)

    Parameters
    ----------
    src : Tensor
        input tensor
    block : dict
        dictionary with entries:
             - <name:str> name of the block.
             - <norm:fun> normalization function to apply. None if none should be applied.
             - <reset:fun> function computing the reset gate. None if none should be applied.
             - <gate:fun> function computing the highway gate. None if none should be applied.
             - <function:fun> actual main block to apply in the middle to compute the output from the input.
    params: dict
        dictionary with entries:
             - <norm_pos:str> normalization function position. Can be one of: ['pre', 'post', 'none'].
             - <reset_gates:bool> flag signaling to use the reset gate.
             - <highway_gates:bool> flag signaling to use the highway gate.
    in_function: function
        function called before executing block['function']. If None, nothing is executed. Default None.
    in_args: dict
        dictionary containing the arguments for in_function. Default {}.
    out_function: function
        function called after executing block['function']. If None, nothing is executed. Default None.
    out_args: dict
        dictionary containing the arguments for out_function. Default {}.

    Returns
    -------
    out : Tensor
        output tensor, computed from applying the block pipeline
    """

    assert params['norm_pos'] in LAYER_NORM_POSITION_MODES, f'norm_pos in {block["name"]} \"{params["norm_pos"]}\" was not implemented, must be one of {LAYER_NORM_POSITION_MODES}'

    ##################### 1 - pre-norm ###############################
    src1 = block['norm'](src) if params['norm_pos'] == LAYER_NORM_PRE else src

    ##################### 2 - input gating ###########################
    if params['reset_gates']:   # 2a - reset gate
        reset = block['reset'](src1)
        src2 = reset * src1
    else:                       # 2b - everything flows through
        src2 = src1

    # ################### 3 - apply input preprocessing ##############
    if in_function is not None:
        src2 = in_function(src2, **in_args)

    # ################### 4 - apply the main block ###################
    if isinstance(src2, dict):
        src3 = block['function'](**src2)
    else:
        src3 = block['function'](src2)

    # ################### 5 - apply output preprocessing #############
    if out_function is not None:
        src3 = out_function(src3, **out_args)

    # ################### 6 - output gating ##########################
    if params['highway_gates']:     # 6a - highway
        gate = block['gate'](src1)
        src = gate * src + (1 - gate) * src3
    else:                           # 6b - residuals
        src = src + src3

    # ################### 7 - post-norm ##############################
    src = block['norm'](src) if params['norm_pos'] == LAYER_NORM_POST else src

    return src

################################################################################################################
############################################## TRANSFORMER BLOCK ###############################################
################################################################################################################

class TransformerLikeConv(nn.Module):
    """ Transformer block using a local self-attention mechanism instead of the usual self-attention.
    This class also includes the possibilities of adding a reset gate (from the Neural GPU https://arxiv.org/abs/1511.08228)
    and an highway gate (https://arxiv.org/abs/1505.00387) to each residual block (self-attention and step function),
    and to change the placement of the layer normalization (can also remove it completely).
    To have the vanilla transformer + local self-attn, just leave default parameters for these three optionals.
    """

    def __init__(self, embedding_size=16, kernel_size=3, groups=1, heads=8, num_units_step=128, dropout=0.1,
                 highway_gates=False, reset_gates=False,
                 norm_pos='post', layer_norm_eps=1e-5, device=None):
        """ Transformer block including the local/conv self-attention, reset gates, highway gates, normalization position.
        The convolutional attention is build having in_channels == out_channels == embedding_size.

        Parameters
        ----------
        embedding_size : int
            size of vectors of the input grid, i.e. the size of single input embedding vectors. Must be divisible by <groups> and <heads>. Default 16.
        kernel_size : int
            width and height of the attention window. Default 3.
        groups : int
            number of groups for the QKV linear projections. See the documentation for AttentionConv/AttentionConvFull for more details about this parameter.
        heads : int
            number of heads when computing attention. When heads==embedding_size, the preferred attention conv will be AttentionConvFull as it is more time-efficient.
            See the documentation for AttentionConv/AttentionConvFull for more details about this parameter.
        num_units_step : int
            number of neurons for the hidden layers of the step function.
        dropout : float
            dropout probability for the step function.
        highway_gates : bool
            if True, add an highway gating system [y = H(x)*x + (1-H(x))*F(x)] in place of the usual residual [y = x + F(x)]. Default False.
        reset_gates : bool
            if True, add a reset gating system which forgets part of the input before applying the blocks. Default False.
        norm_pos : str
            normalization position. Can be one of ['pre', 'post', 'none'], where 'pre' applies it to the input (before reset gating),
            'post' applies it to the output (after residuals as usual in transformers) and 'none' does not apply any normalization. Default 'post'.
        layer_norm_eps : float
            parameter epsilon of layer normalization. Default 1e-5.
        device : None
            computing device to be used.
        """

        super(TransformerLikeConv, self).__init__()

        #################### MAIN PARAMETERS ####################
        self.embedding_size = embedding_size

        ################ LOCAL ATTENTION BLOCK BUILDING ################

        # MODE 1: use AttentionConvFull in the case heads == embedding_size as it is more efficient
        if heads == embedding_size:
            print('local attention mode 1: heads == embedding_size')
            attn_conv = AttentionConvFull(
                in_channels=embedding_size, out_channels=embedding_size,
                kernel_size=kernel_size, groups=groups,
                device=device
            )
        # MODE 2: use AttentionConv for the general case
        else:
            print('local attention mode 2: general case')
            attn_conv = AttentionConv(
                in_channels=embedding_size, out_channels=embedding_size,
                kernel_size=kernel_size, groups=groups, heads=heads,
                device=device
            )
        self.attn_conv = attn_conv

        ################ STEP FUNCTION BLOCK BUILDING ################

        self.step = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_size, num_units_step),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(num_units_step, num_units_step),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(num_units_step, embedding_size)
        )

        ######################### PACKAGE THE TWO BLOCKS #########################
        ###### 1 - prepare vanilla blocks (without optionals) ####################
        self.sasa_block = {
            'name': 'sasa',
            'function': self.attn_conv
        }
        self.step_block = {
            'name': 'step',
            'function': self.step
        }

        ###### 2 -  add optionals ################################################
        # 2a - add layer normalization
        assert norm_pos in LAYER_NORM_POSITION_MODES, f'Norm position {norm_pos} not implemented, must be one of {LAYER_NORM_POSITION_MODES}'
        self.norm_pos = norm_pos

        if norm_pos != LAYER_NORM_NONE:
            self.norm_sasa = nn.LayerNorm(normalized_shape=embedding_size, eps=layer_norm_eps)
            self.norm_step = nn.LayerNorm(normalized_shape=embedding_size, eps=layer_norm_eps)

            self.sasa_block['norm'] = self.norm_sasa
            self.step_block['norm'] = self.norm_step

        # 2b - add highway optional
        self.has_highway_gates = highway_gates
        if highway_gates:
            # build shallow gates
            self.gate_sasa = nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.Sigmoid())
            self.gate_step = nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.Sigmoid())

            self.sasa_block['gate'] = self.gate_sasa
            self.step_block['gate'] = self.gate_step

        # 2c - add reset optional
        self.has_reset_gates = reset_gates
        if reset_gates:
            # build shallow gates
            self.reset_sasa = nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.Sigmoid())
            self.reset_step = nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.Sigmoid())

            self.sasa_block['reset'] = self.reset_sasa
            self.step_block['reset'] = self.reset_step

        ################ PACKAGE PARAMETERS AND FLAGS ################
        self.params = {
            'norm_pos': norm_pos,
            'reset_gates': reset_gates,
            'highway_gates': highway_gates
        }

    def forward(self, src):
        """ Pass an input through the SASA and STEP blocks of a transformer.

        Parameters
        ----------
        src : Tensor
            input tensor of shape (B,H,W,E), where:
                - B: batch size
                - H: height of the grid
                - W: width of the grid
                - E: embedding size

        Returns
        -------
        out : Tensor
            output tensor of shape (B,H,W,E) (same shape as the input)
        """

        # COMPUTE SASA
        src = apply_block(src, self.sasa_block, self.params)

        # COMPUTE STEP
        src = apply_block(src, self.step_block, self.params)

        return src