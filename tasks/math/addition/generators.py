from torch.utils.data import Dataset
from ..alphabets import generate_arithmetic_alphabet, PAD_SYMB, BASE_CONVERSIONS, BASE_DIGITS
from tasks.utils.tokenizer import pad_batch, vocab_alphabet2index, index_tokenize, tokenizer_rich_str2chars

import random
import numpy as np



BIN_ADDITION_ALPHA = generate_arithmetic_alphabet(
	['+', '='], [PAD_SYMB], base=2)
DEC_ADDITION_ALPHA = generate_arithmetic_alphabet(
	['+', '='], [PAD_SYMB], base=10)

BIN_ADDITION_VOCAB = vocab_alphabet2index(BIN_ADDITION_ALPHA)
DEC_ADDITION_VOCAB = vocab_alphabet2index(DEC_ADDITION_ALPHA)

BASE_ADDITION_VOCABS = {
	10: DEC_ADDITION_VOCAB,
	2: BIN_ADDITION_VOCAB
}

def get_expected_vocab_size(numeric_base=10):
	return len(BASE_ADDITION_VOCABS[numeric_base])

def get_expected_pad_index(numeric_base=10):
	return BASE_ADDITION_VOCABS[PAD_SYMB]

def _get_ranges(tuple_range, groups):
	""" Separate a range/interval, say [A, B], in <groups> equal parts
	"""
	slice_size = (tuple_range[1] - tuple_range[0]) / groups
	assert slice_size >= 1, 'The given data range is smaller than the number of groups. Try to lower the number of groups'
	ranges = [(tuple_range[0]+int(i*slice_size), tuple_range[0]+int((i+1)*slice_size)) for i in range(groups)]
	return ranges

class BatchedSumSeqDataset(Dataset):

	def __init__(self, operands_num, operands_size, batches_num, batches_size,
				 num_groups=1, size_groups=1, result_pad_symb=PAD_SYMB, numeric_base=10,
				 didactic_gen_collection=None, use_didactic_samples_prob=0.0):
		"""
		Parameters
		----------
		operands_num : int | tuple[int] | list[int]
			- if int, fixed number of operands for all generated operations.
			- if tuple[int] or list[int], range for sampling random numbers of operands for each operation. Extremes are included.
		operands_size : int | tuple[int] | list[int]
			- if int, fixed size of each operand for all generated operations.
			- if tuple[int] or list[int], range for sampling random sizes of operands for each operation. Extremes are included.
		batches_num : int
			size of dataset. Only used when iterating over this dataset. Calling len will return this exact number.
		batches_size : int
			number of examples to generate for each batch.
		num_groups : int
			number of groups for which to divide the operands_num range. This is useful for increasing space efficiency in returned tensors. Default 1.
		size_groups : int
			number of groups for which to divide the operands_size range. This is useful for increasing space efficiency in returned tensors. Default 1.
		result_pad_symb : str
			symbol to use for padding purposes. This special symbol is appended to build tensors out of different lenghts sequences of symbols.
			It must be included in the specified base alphabet. Default PAD_SYMB
		numeric_base : int
			base of numbers to be generated. Default 10.
		didactic_gen_collection : GeneratorsCollection, optional
			collection of didactic examples generators. It can be used to generate structured examples, defined through a pattern. Default None.
		use_didactic_samples_prob : float
			probability of generating a didactic example instead of a regular one. Default 0.0.
		"""

		if not (isinstance(operands_size, list) or isinstance(operands_size, tuple)):
			operands_size = (operands_size, operands_size)

		if not (isinstance(operands_num, list) or isinstance(operands_num, tuple)):
			operands_num = (operands_num, operands_num)

		# operands parameters, increase max because 
		self.operands_num = (operands_num[0], operands_num[1]+1)
		self.operands_size = (operands_size[0], operands_size[1]+1)
		# batched examples parameters
		self.batches_num = batches_num
		self.batches_size = batches_size
		# groups of examples parameters
		self.num_groups = num_groups    # groups on operands number
		self.size_groups = size_groups  # groups on operands size
		# separate both the number and size of operands into groups
		self.num_ranges = _get_ranges(self.operands_num, self.num_groups)
		self.size_ranges = _get_ranges(self.operands_size, self.size_groups)

		# set others
		self.result_pad_symb = result_pad_symb
		self.didactic_gen_collection = didactic_gen_collection
		self.use_didactic_samples_prob = use_didactic_samples_prob

		# select base
		self.base =       numeric_base
		self.conversion = BASE_CONVERSIONS[numeric_base]
		self.digits =     BASE_DIGITS[numeric_base]
		self.vocab =      BASE_ADDITION_VOCABS[numeric_base]

	def get_pad_index(self):
		return self.vocab[self.result_pad_symb]

	def get_vocab_size(self):
		return len(self.vocab)

	def __len__(self):
		return self.batches_num

	def __getitem__(self, idx):

		# initialize division values
		tot_bsize = self.batches_size
		tot_groups = self.size_groups * self.num_groups

		batches = []

		for n_range in self.num_ranges:
			for s_range in self.size_ranges:

				curr_bsize = tot_bsize // tot_groups
				tot_bsize -= curr_bsize
				tot_groups -= 1

				curr_seqs = []
				curr_results = []

				for i in range(curr_bsize):

					sequence = None

					# with probability generate a didactic sample
					if self.use_didactic_samples_prob > random.random() and self.didactic_gen_collection is not None:
						sequence = self.didactic_gen_collection(
							n_range, s_range)
					# else generate a random sequence of numbers
					if sequence is None:
						sequence = []
						# compute division operands num
						curr_opsnum = np.random.randint(*n_range)

						for i in range(curr_opsnum):
							curr_numsize = np.random.randint(*s_range)
							curr_operand = ''.join(
								np.random.choice(self.digits, curr_numsize))
							sequence.append(curr_operand)

					result = 0
					for s in sequence:
						result += int(s, self.base)

					# tokenize sequence
					sequence = '+'.join(sequence)
					sequence += '='
					sequence = index_tokenize(sequence, self.vocab, tokenizer_rich_str2chars)
					curr_seqs.append(sequence)
					# tokenize result
					result = self.conversion(result)
					result = index_tokenize(result, self.vocab, tokenizer_rich_str2chars)
					result = result
					curr_results.append(result)
				# end batch for

				# generate division batch
				curr_seqs =     pad_batch(curr_seqs, self.vocab[self.result_pad_symb], where='after')
				curr_results =  pad_batch(curr_results, self.vocab[self.result_pad_symb], where='before')

				batches.append({'seq': curr_seqs, 'res': curr_results,
							   'max_opsnum': n_range[1], 'max_opsize': s_range[1]})

		return (batches)
