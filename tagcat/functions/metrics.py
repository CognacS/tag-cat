import torch

# RETURN MATCHES AND CORRECT MATCHES
def compute_matches_samplewise(prediction, target, compute_argmax=True, ignore_symb=0, ignore_symb_matchings=True, ignore_symb_is_ambiguous=False, return_valid_matches=False):
	""" Return the correct and valid matches per sample using the strict accuracy definition: compute accuracy as usual
	but ignore matches between prediction and target where they both present the <ignore_symb>. This drastically reduces the number
	of correct matches when accuracy is inflated by trivial predictions, such as PAD-PAD.
		
	Parameters
	----------
	prediction : Tensor
		tensor of shape (B, W, C) of probabilities or shape (B, W) of classes, where:
			- B : batch size
			- W : output sequence size
			- C : number of classes
	target : Tensor
		tensor of shape (B, W) of ground truth classes to compare with prediction
	compute_argmax : bool
		flag to compute the argmax value if prediction is a tensor of probabilities. Default True.
	ignore_symb : int
		index of a symbol for some alphabet which should be ignored in double-matchings, such as the the <PAD> symbol. Default 0.
	ignore_symb_matchings : bool
		flag for ignoring the ignore_symb double-matchings. If True this is the strict accuracy. If False, this is standard accuracy. Default True.
	ignore_symb_is_ambiguous : bool
		flag for informing this function that the <ignore_symb> is not a special symbol, and may be used for the solution. Setting this flag to True
		can help when the ignore_symb is set to "0" instead of "<PAD>". In this case, ignored matches start from the last.
	return_valid_matches : bool
		additionaly return the number of valid matches for each 
	Returns
	-------
	correct_matches : Tensor
		tensor of shape (B,W) having 1s where matches are correct AND valid, and 0s where matches are not correct or not valid.
	valid_matches : Tensor
		tensor of shape (B,W) having 1s where matches are valid (as to the strict accuracy definition), and 0s where matches are not valid.
		This will be an array of 1s if ignore_symb_matching=False.
	"""
	# compute argmax if needed
	pred = torch.argmax(prediction, dim=2) if compute_argmax else prediction

	if ignore_symb_matchings:
		# compute masks:
		# - 0 if the symbol is the ignore_symbol
		# - 1 if the symbol is not the ignore_symbol

		if ignore_symb_is_ambiguous:
			# observe where the ignored symbols stops being used as padding
			# will have zeroes until any other symbol is used, then it will never be zero again
			mask_pred =     torch.cumsum(pred != ignore_symb, dim=1) > 0
			mask_target =   torch.cumsum(target != ignore_symb, dim=1) > 0

		else:
			# just observe where the symbol is not used
			mask_pred =     pred != ignore_symb
			mask_target =   target != ignore_symb
		
		# compute valid matches as an OR between 
		valid_matches = torch.any(torch.stack([mask_pred, mask_target]), axis=0)

		# set all last elements valid matches, because at least one element must be considered
		valid_matches[:, -1] = 1

	else:
		# standard accuracy takes into account all matches
		valid_matches = torch.ones_like(target)

	correct_matches = torch.all(torch.stack([valid_matches, pred == target]), axis=0)
	return  correct_matches, valid_matches

# SAMPLE-WISE CHARACTER LEVEL ACCURACY
def compute_char_acc_samplewise(output, target, compute_argmax=True, ignore_symb=0, ignore_symb_matchings=True, ignore_symb_is_ambiguous=False):
	""" Compute the character level accuracy sample-wise using the strict accuracy definition: compute accuracy as usual
	but ignore matches between prediction and target where they both present the <ignore_symb>. This drastically reduces the number
	of correct matches when accuracy is inflated by trivial predictions, such as PAD-PAD.
		
	Parameters
	----------
	prediction : Tensor
		tensor of shape (B, W, C) of probabilities or shape (B, W) of classes, where:
			- B : batch size
			- W : output sequence size
			- C : number of classes
	target : Tensor
		tensor of shape (B, W) of ground truth classes to compare with prediction
	compute_argmax : bool
		flag to compute the argmax value if prediction is a tensor of probabilities. Default True.
	ignore_symb : int
		index of a symbol for some alphabet which should be ignored in double-matchings, such as the the <PAD> symbol. Default 0.
	ignore_symb_matchings : bool
		flag for ignoring the ignore_symb double-matchings. If True this is the strict accuracy. If False, this is standard accuracy. Default True.
	ignore_symb_is_ambiguous : bool
		flag for informing this function that the <ignore_symb> is not a special symbol, and may be used for the solution. Setting this flag to True
		can help when the ignore_symb is set to "0" instead of "<PAD>". In this case, ignored matches start from the last.

	Returns
	-------
	char_accuracy_samplewise : Tensor
		tensor of shape (B,) holding the char-accuracy of each sample in the batch
	"""
	correct_matches, valid_matches = compute_matches_samplewise(
		output, target,
		compute_argmax=compute_argmax,
		ignore_symb=ignore_symb,
		ignore_symb_matchings=ignore_symb_matchings,
		ignore_symb_is_ambiguous=ignore_symb_is_ambiguous,
		return_valid_matches=True
	)
	return torch.sum(correct_matches, dim=1) / torch.sum(valid_matches, dim=1)


# TOTAL CHARACTER LEVEL ACCURACY 
def compute_char_acc(output, target, compute_argmax=True, ignore_symb=0, ignore_symb_matchings=True, ignore_symb_is_ambiguous=False):
	""" Compute the character level accuracy on the whole batch using the strict accuracy definition: compute accuracy as usual
	but ignore matches between prediction and target where they both present the <ignore_symb>. This drastically reduces the number
	of correct matches when accuracy is inflated by trivial predictions, such as PAD-PAD.
		
	Parameters
	----------
	prediction : Tensor
		tensor of shape (B, W, C) of probabilities or shape (B, W) of classes, where:
			- B : batch size
			- W : output sequence size
			- C : number of classes
	target : Tensor
		tensor of shape (B, W) of ground truth classes to compare with prediction
	compute_argmax : bool
		flag to compute the argmax value if prediction is a tensor of probabilities. Default True.
	ignore_symb : int
		index of a symbol for some alphabet which should be ignored in double-matchings, such as the the <PAD> symbol. Default 0.
	ignore_symb_matchings : bool
		flag for ignoring the ignore_symb double-matchings. If True this is the strict accuracy. If False, this is standard accuracy. Default True.
	ignore_symb_is_ambiguous : bool
		flag for informing this function that the <ignore_symb> is not a special symbol, and may be used for the solution. Setting this flag to True
		can help when the ignore_symb is set to "0" instead of "<PAD>". In this case, ignored matches start from the last.

	Returns
	-------
	char_accuracy_samplewise : Tensor
		1 element tensor holding the char-level accuracy counting all valid matches together.
	"""
	correct_matches, valid_matches = compute_matches_samplewise(
		output, target,
		compute_argmax=compute_argmax,
		ignore_symb=ignore_symb,
		ignore_symb_matchings=ignore_symb_matchings,
		ignore_symb_is_ambiguous=ignore_symb_is_ambiguous,
		return_valid_matches=True
	)
	return (torch.sum(correct_matches) / torch.sum(valid_matches)).mean()

# TOTAL SEQUENCE LEVEL ACCURACY 
def compute_seq_acc(output, target, compute_argmax=True, ignore_symb=0, ignore_symb_matchings=True, ignore_symb_is_ambiguous=False):
	""" Compute the sequence level accuracy on the whole batch using the strict accuracy definition: compute accuracy as usual
	but ignore matches between prediction and target where they both present the <ignore_symb>. This drastically reduces the number
	of correct matches when accuracy is inflated by trivial predictions, such as PAD-PAD.
		
	Parameters
	----------
	prediction : Tensor
		tensor of shape (B, W, C) of probabilities or shape (B, W) of classes, where:
			- B : batch size
			- W : output sequence size
			- C : number of classes
	target : Tensor
		tensor of shape (B, W) of ground truth classes to compare with prediction
	compute_argmax : bool
		flag to compute the argmax value if prediction is a tensor of probabilities. Default True.
	ignore_symb : int
		index of a symbol for some alphabet which should be ignored in double-matchings, such as the the <PAD> symbol. Default 0.
	ignore_symb_matchings : bool
		flag for ignoring the ignore_symb double-matchings. If True this is the strict accuracy. If False, this is standard accuracy. Default True.
	ignore_symb_is_ambiguous : bool
		flag for informing this function that the <ignore_symb> is not a special symbol, and may be used for the solution. Setting this flag to True
		can help when the ignore_symb is set to "0" instead of "<PAD>". In this case, ignored matches start from the last.

	Returns
	-------
	seq_accuracy_samplewise : Tensor
		1 element tensor holding the seq-level accuracy, counting samples which have a perfect char accuracy of 1.0 as correct samples, while others are errors.
	"""
	char_accs_samples = compute_char_acc_samplewise(
		output, target,
		compute_argmax=compute_argmax,
		ignore_symb=ignore_symb,
		ignore_symb_matchings=ignore_symb_matchings,
		ignore_symb_is_ambiguous=ignore_symb_is_ambiguous
	)
	return torch.sum(char_accs_samples == 1) / char_accs_samples.shape[0]
