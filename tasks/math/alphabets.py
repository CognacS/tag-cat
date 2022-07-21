################################################################################################################
############################################### SPECIAL SYMBOLS ################################################
################################################################################################################

PAD_SYMB = '<PAD>'

################################################################################################################
############################################# ALPHABET GENERATORS ##############################################
################################################################################################################

def generate_digits_alphabet(base=10):
    # build digits
    digits = [str(i) for i in range(min(base, 10))]
    if base > 10:
        digits += [chr(ord('A') + i) for i in range(base - 10)]

    return digits


def generate_arithmetic_alphabet(operations, special_symbols, base=10):

    digits = generate_digits_alphabet(base)

    return digits + operations + special_symbols


def generate_math_aphabet(literals, operations, special_symbols):
    return literals + operations + special_symbols


################################################################################################################
######################################## ANY BASE INT-TO-STR FUNCTIONS #########################################
################################################################################################################
def int_str(x):
  return str(x)

def bin_str(x):
  return bin(x)[2:]

DEC_DIGITS = generate_digits_alphabet(base=10)
BIN_DIGITS = generate_digits_alphabet(base=2)

BASE_CONVERSIONS = {10: int_str, 2: bin_str}
BASE_DIGITS = {10: DEC_DIGITS, 2: BIN_DIGITS}