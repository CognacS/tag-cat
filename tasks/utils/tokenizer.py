import collections.abc

generic_sequence = collections.abc.Sequence

################################################################################################################
######################################### VOCABULARY BUILDING FUNCTIONS ########################################
################################################################################################################

# symbol -> index
def vocab_alphabet2index(alphabet):
    """ Returns a dictionary (vocabulary) mapping each alphabet symbol to an index starting from 0 to len(alphabet). The ordering is the same as in alphabet.

    Parameters
    ----------
    alphabet : list[symb_type]
        sequence of symbols
    
    Returns
    -------
    vocab : dict[symb_type, int]
        dictionary mapping each alphabet symbol to an index in [0, len(alphabet)-1]
    """
    return {t:i for i, t in enumerate(alphabet)}

# index -> symbol
def vocab_index2alphabet(alphabet):
    """ Returns a dictionary mapping each index starting from 0 to len(alphabet) to an alphabet symbol. The ordering is the same as in alphabet.

    Parameters
    ----------
    alphabet : list[symb_type]
        sequence of symbols
    
    Returns
    -------
    vocab : dict[int, symb_type]
        dictionary mapping each index in [0, len(alphabet)-1] to an alphabet symbol
    """
    return {i:t for i, t in enumerate(alphabet)}

# reverse -> ===> <-
def vocab_reverse_map(vocab):
    """ Returns the reverse map of a vocabulary.

    Parameters
    ----------
    vocab : dict[symb_type, int] | dict[int, symb_type]
        dictionary mapping each alphabet symbol to an index in [0, len(alphabet)-1] or the inverse
    
    Returns
    -------
    vocab_out : dict[int, symb_type] | dict[symb_type, int]
        the inverse mapping of the input vocabulary
    """
    return {v:k for k, v in vocab.items()}


# sequence of [any] -(tokenizer_function)-> sequence of [tokens] -(vocab_alpha2index)-> sequence of [indices]
def index_tokenize(sequence, vocab_alpha2index, tokenizer_function, **tokenizer_arguments):
    """ Break down the sequence with a tokenizer function and replace each symbol with the corresponding entry of the vocabulary

    Parameters
    ----------
    sequence : generic_sequence
        sequence of any type, that can be broken into symbols by the tokenizer_function
    vocab_alpha2index : dict[symb_type, int]
        dictionary mapping each alphabet symbol to an index in [0, len(alphabet)-1]
    tokenizer_function : function(generic_sequence) -> list[symb_type]
        function breaking sequence into a list of symbols
    
    Returns
    -------
    indices : list[int]
        sequence of integer indices where each element has been mapped from a symbol composing sequence (according to the tokenizer_function)
    """
    symbols = tokenizer_function(sequence, **tokenizer_arguments)
    indices = [vocab_alpha2index[s] for s in symbols]
    return indices


# sequence of [indices] -(vocab_index2alphavocab_alpha2index)-> sequence of [tokens] -(detokenizer_function)-> sequence of [any]
def index_detokenize(indices, vocab_index2alpha, detokenizer_function, **detokenizer_arguments):
    """ Rebuild the original sequence by replacing each index with the corresponding symbol of the vocabulary, and reconstruct with a detokenizer function

    Parameters
    ----------
    indices : list[int]
        sequence of integer indices where each element can be mapped to a symbol from a vocabulary
    vocab_index2alpha : dict[int, symb_type]
        dictionary mapping each index in [0, len(alphabet)-1] to an alphabet symbol
    detokenizer_function : function(list[symb_type]) -> generic_sequence
        function building a sequence from a list of symbols
    
    Returns
    -------
    sequence : generic_sequence
        sequence of any type, reconstructed through a vocabulary and a detokenizer_function
    """
    symbols = [vocab_index2alpha[i] for i in indices]
    sequence = detokenizer_function(symbols, **detokenizer_arguments)
    return sequence


################################################################################################################
############################################# TOKENIZER FUNCTIONS ##############################################
################################################################################################################

class TokenizationException(Exception):
    pass

def extract_special_symbols(sequence):
    """ Break down a sequence of characters into ordinary strings and special symbols, i.e. strings enclosed in brackes.
    E.g.
    Input: "Hi, my name is <PLACEHOLDER>, nice to meet you."
    Output: "Hi, my name is ", "<PLACEHOLDER>", ", nice to meet you.", where "<PLACEHOLDER>" is flagged as a special symbol.

    Parameters
    ----------
    sequence : str
        sequence of characters containing special symbols, i.e. strings enclosed in brackets.
    
    Returns
    -------
    final_parts: list[str]
        broken down pieces of the input sequence, where an element can be an ordinary string, or a special symbol (also a string enclosed by <>).
    is_special : list[bool]
        flags indicating whether each element of final_parts is a special symbol or not.
    """

    final_parts =  []
    is_special = []

    # split by opening bracket
    parts = sequence.split('<')

    # check first broken down part (should not have > brackets)
    if parts[0].find('>') >= 0:
        raise TokenizationException('found a > bracket with no corresponding opening bracket <')
    if len(parts[0]) > 0:
        final_parts.append(parts[0])
        is_special.append(False)

    # check other broken parts
    for p in parts[1:]:
        # find closing brackets in a part
        subparts = p.split('>')
        # a correct example is when there are 2 resulting subparts from the split
        # even if > is the terminating symbol, split will return 2 elements, where the second element is an empty string
        if len(subparts) > 2:
            raise TokenizationException('found a > bracket with no corresponding opening bracket <')
        elif len(subparts) == 1:
            raise TokenizationException('found a < bracket with no corresponding closing bracket >')
        # no errors: append the special symbol
        final_parts.append(f'<{subparts[0]}>')
        is_special.append(True)
        # if the second part is a non-empty string, include it
        if len(subparts[1]) > 0:
            final_parts.append(subparts[1])
            is_special.append(False)
    return final_parts, is_special

def tokenizer_str2chars(sequence):
    """ Simple tokenizer breaking a string into raw characters

    Parameters
    ----------
    sequence : str
        sequence of characters
    
    Returns
    -------
    final_parts: list[str]
        sequence of tokens, in this case characters
    """
    return list(sequence)

def tokenizer_rich_str2chars(sequence):
    """ Rich tokenizer, breaking a string into raw characters and special symbols, enclosed inside brackets <>.
    E.g.
    Input: "Hi, my name is <PLACEHOLDER>"
    Output: "H", "i", ",", "m", "y", " ", "n", "a", "m", "e", " ", "i", "s", " ", "<PLACEHOLDER>"

    Parameters
    ----------
    sequence : str
        sequence of characters, also containing special symbols.
    
    Returns
    -------
    final_parts: list[str]
        sequence of tokens, in this case characters and special symbols
    """
    parts, is_special_list = extract_special_symbols(sequence)
    final_parts = []
    
    for p, is_special in zip(parts, is_special_list):
        if is_special:
            final_parts += [p]
        else:
            final_parts += tokenizer_str2chars(p)
    
    return final_parts

################################################################################################################
############################################ DETOKENIZER FUNCTIONS #############################################
################################################################################################################

class DetokenizationException(Exception):
    pass

def is_special_symbol(symbol):
    return symbol[0] == '<' and symbol[-1] == '>' and all([s not in symbol[1:-1] for s in ['<','>']])

def detokenizer_chars2str(tokens):
    """ Simple detokenizer joining raw characters into a string

    Parameters
    ----------
    tokens : list[str]
        list of string tokens
    
    Returns
    -------
    final_sequence: string
        string of the joined tokens
    """
    return ''.join(tokens)

def detokenizer_rich_chars2str(tokens, specialsymb_behavior='remove'):
    """ Rich detokenizer joining raw characters into a string, and defining behaviors for special symbols

    Parameters
    ----------
    tokens : list[str]
        list of string tokens
    specialsymb_behavior : str | dict[str: (str)->str]
        string defining the common behavior for all symbols. Can be 'remove' to remove all special symbols, or 'keep' to keep symbols.
        Can also define a dictionary mapping each special symbols to a behavior function taking as input the entire string obtained up
        to the encounter with the special symbol, and returning as output a new string to replace the input string.
    Returns
    -------
    final_sequence: string
        string of the joined tokens
    """
    final_sequence = ''
    for token in tokens:
        # TOKEN IS SPECIAL SYMBOL
        if is_special_symbol(token):
            # BEHAVIOR IS STRING
            if isinstance(specialsymb_behavior, str):
                if specialsymb_behavior == 'keep':
                    final_sequence += token
                elif specialsymb_behavior == 'remove':
                    continue
                else:
                    raise DetokenizationException(f'given an unknown special symbol behavior: {specialsymb_behavior}')
            # BEHAVIOR IS DICT
            elif isinstance(specialsymb_behavior, dict):
                try:
                    behavior = specialsymb_behavior[token]
                except KeyError as err:
                    raise DetokenizationException(f'special symbol {token} is not included in the given map of behaviors')

                final_sequence = behavior(final_sequence)
            else:
                raise DetokenizationException(f'given special symbol behavior is neither a string nor a dictionary')
        # TOKEN IS A STANDARD SYMBOL
        else:
            final_sequence += token
            
    return final_sequence


################################################################################################################
######################################### TENSORS AUXILIARY FUNCTIONS ##########################################
################################################################################################################

import numpy as np
import torch

class IndexPadException(Exception):
    pass

def pad_to_size(x, final_size, pad_value=0, where='after'):
    """ Pad <pad_value> to array x in position <where> in order to reach <final_size>

    Parameters
    ----------
    x : 1D array of integers
        array containing symbol indices
    final_size : int
        size of the resulting array x
    pad_value : int
        value to pad on the input sequence x. Default 0
    where : str
        position where to pad <pad_value>. If where='after', it will be inserted as last elements. If where='before', it will be inserted as first elements. Default 'after'.
    Returns
    -------
    padded_x : 1D array of integers
        x padded before or after with <pad_value> up to size <final_size>
    """
    if len(x) > final_size:
        raise IndexPadException(f'given final_size ({final_size}) is smaller than the size of the given array x ({len(x)})')
    if where == 'after':
        pad_width = ((0, final_size - len(x)))
    elif where == 'before':
        pad_width = ((final_size - len(x), 0))
    else:
        raise IndexPadException(f'where clause must be \'after\' or \'before\'. Given: {where}')

    return np.pad(x, pad_width=pad_width, constant_values=pad_value)


def pad_batch(batch_x, pad_value_x=0, where='after'):
    """ Pad <pad_value> to all arrays of batch_x, in order to build a tensor.

    Parameters
    ----------
    batch_x : list of 1D arrays of integers
        list of B index symbols sequences, where each can have an arbitrary length Li
    pad_value : int
        value to pad on the input sequences x. Default 0
    where : str
        position where to pad <pad_value>. If where='after', it will be inserted as last elements. If where='before', it will be inserted as first elements. Default 'after'.

    Returns
    -------
    tensor_x : tensor
        batch tensor of shape (B, max(Li))
    """
    # compute maximum length in the batch
    max_len = max([len(x) for x in batch_x])

    # for each element, pad to max length, in order to build a tensor
    tensor_x = torch.tensor(np.stack([
        pad_to_size(x, final_size=max_len, pad_value=pad_value_x, where=where) for x in batch_x
    ]))
    
    return tensor_x

