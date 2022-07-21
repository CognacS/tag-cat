import torch
import torch.functional as F


def get_class_weightings_pad(num_classes, pad_class, pad_weight=0.1, device=None):
	weights = torch.ones(num_classes, device=device)
	weights[pad_class] = pad_weight
	return weights


################################################################################################################
####################################### CUSTOM LOSS FOR WEIGHTING EXAMPLES #####################################
################################################################################################################

class SampleWeightedLoss:
	def __init__(self, loss_fn):
		self.loss_fn = loss_fn

	def __call__(self, input, target, weights_per_sample=None, normalize_with_weights=False):
		"""Computes the given loss function by weighting each sample with <weights_per_sample>. If it is None, returns the default averaged loss.

		Args:
			input : Tensor
				input tensor of shape (B, *, C) where * can be any number of dimensions.
			target : Tensor
				target tensor of shape (B, *) where * can be any number of dimensions.
			weights_per_sample : Tensor
				weights tensor of shape (B,). Defaults to None.

		Returns:
			_type_: _description_
		"""

		# move last dimension (classes dim) as 2nd dimension (index 1)
		shape = input.shape
		ndims = len(shape)
		perm = [0, ndims-1] + list(range(1, ndims-1))
		input = input.permute(perm)

		# if no sample-wise weights -> simply call loss
		if weights_per_sample is None:
			return self.loss_fn(input, target)

		# if using sample-wise weights
		else:
			# save reduction type and update to none
			saved_reduction = self.loss_fn.reduction
			self.loss_fn.reduction = 'none'

			# compute unreduced losses
			losses = self.loss_fn(input, target)
			# restore reduction type
			self.loss_fn.reduction = saved_reduction

			# reduce over all dimensions but the batch one
			losses = losses.sum(dim=list(range(1, losses.ndimension())))

			# compute normalizing term
			if self.loss_fn.weight is not None:
				total = self.loss_fn.weight[target].sum(dim=list(range(1, target.ndimension())))
			else:
				total = target.shape[1:].numel()

			if normalize_with_weights:
				norm_term = (weights_per_sample * total).sum()
			else:
				norm_term = total.sum()

			# if there are leading dimensions, remove them
			weights_per_sample = weights_per_sample.view(len(weights_per_sample))

			"""
			# reshape weights_per_sample to be broadcasted on the loss
			ones_dims = [1] * (target.ndimension-1) # ones over all dimensions but the batch(first) one
			new_shape = [len(weights_per_sample)] + ones_dims
			weights_per_sample = weights_per_sample.view(new_shape)
			"""

			# reduce the loss over the batches
			loss = (weights_per_sample * losses).sum() / norm_term

			return loss


################################################################################################################
################################### GEOMETRIC DISTRIBUTION AUXILIARY FUNCTIONS #################################
################################################################################################################

def pad_to(t, padding, dim=-1, value=0.):
	if dim > 0:
		dim = dim - t.ndim
	zeroes = -dim - 1
	return F.pad(t, (*((0, 0) * zeroes), *padding), value=value)


def exclusive_cumprod(t, dim=-1):
	cum_prod = t.cumprod(dim=dim)
	return pad_to(cum_prod, (1, -1), value=1., dim=dim)


def calc_geometric(N, l, dim=-1):
	if isinstance(l, torch.Tensor):
		l = l.item()
	t = torch.full((N,), 1 - l)
	return exclusive_cumprod(t, dim=dim) * l

################################################################################################################
################################################### STATISTICS #################################################
################################################################################################################

def bdot_mulsum(a, b):
	# a: (B, V)
	# b: (B, V)
	# ret: (B, 1)
	return (a * b).sum(-1, keepdim=True)

def bdot_bmm(a, b):
	# a: (B, V)
	# b: (B, V)
	# ret: (B, 1)
	return torch.bmm(a.unsqueeze(-2), b.unsqueeze(-1)).squeeze(-1)

bdot = bdot_mulsum

def expected_value(distr, values):
	""" Compute the expected value for each sample in a batch of size B. Each sample has V values, each appearing with its probability.
	This is done by computing the batched dot-product of distr-values vectors

	Parameters
	----------
	distr: Tensor
		tensor of probabilities of shape (B, V), one for each value in <values>
	values: Tensor
		tensor of values to take the expected value over, distributed as <distr>, of shape (B, V)

	Returns
	-------
	expected_value : Tensor
		tensor of shape (B, 1) containing expected values
	"""
	return bdot(distr, values)


################################################################################################################
###################################### KL-DIVERGENCE VARIANTS FOR PONDERNET ####################################
################################################################################################################

# D(pn||pG(lp)) = -H(pn) - log(lp) - log(1-lp) * E_pn[N]
# D(pn||pG(lp)) = E_pn[log(pn)-log(1-lp)n] - log(lp) (actually used, more efficient)


# compute geometric distribution and manually compute KL_div
def distribution_kldiv_pondernet(pn, lambda_p, tol=1e-6):
	""" Computes the original pondernet kldiv by building a geometric distribution and calling the kl_div
	loss function.
	The kl-div is computed between the halting probability and a geometric distribution to regularize it.
	This function does build the geometric distribution, to it may take longer.

	Parameters
	----------
	pn : Tensor
		tensor of halting probabilities of shape (B, N)
	lambda_p : float
		parameter of the geometric distribution
	tol : float
		tolerance value to avoid log(0) cases
	
	Returns
	-------
	divergence : Tensor
		tensor of shape (1,) of the reduced kl-divergence
	"""
	B = pn.shape[0]
	N = pn.shape[-1]
	geom = calc_geometric(N, lambda_p).unsqueeze(0).expand(B, -1).to(pn.device)
	return F.kl_div(torch.log(geom + tol), pn, reduction='batchmean')


def closed_form_kldiv_pondernet(pn, lambda_p, tol=1e-6):
	""" Computes the original pondernet kldiv using the closed form formula:

		D(pn||pG(lambda_p)) = -H(pn) - log(1-lambda_p)*E_pn[n] - log(lambda_p)

	but actually uses:

		D(pn||pG(lambda_p)) = E_pn[log(p_n)-log(1-lambda_p)*n] - log(lambda_p)

	as it is faster to compute.
	The kl-div is computed between the halting probability and a geometric distribution to regularize it.
	This function does not build the geometric distribution, therefore it is faster.

	Parameters
	----------
	pn : Tensor
		tensor of halting probabilities of shape (B, N)
	lambda_p : float
		parameter of the geometric distribution
	tol : float
		tolerance value to avoid log(0) cases
	
	Returns
	-------
	divergence : Tensor
		tensor of shape (1,) of the reduced kl-divergence
	"""
	device = pn.device
	N = pn.shape[-1]
	values = torch.log(pn + tol) - (torch.log(1. - lambda_p + tol) * torch.arange(N, dtype=torch.float, device=device))
	values = expected_value(pn, values)
	divergence = - torch.log(lambda_p + tol) + values
	return divergence.mean()


# EXPERIMENTAL, NOT USED
def closed_form_reg(pn, lambda_p, acc, snap):
	device = pn.device
	a = (1-snap)*acc/(snap + (1-2*snap)*acc)
	N = pn.shape[-1]
	#values = (1-a) * torch.log(pn + 1e-6) + (a - (1-a) * torch.log(1. - lambda_p + 1e-6)) * torch.arange(N, dtype=torch.float, device=device)
	values = (1-a) * torch.log(pn+1e-6) - torch.log(1. - lambda_p +
													1e-6) * torch.arange(N, dtype=torch.float, device=device)

	values = expected_value(pn, values)
	divergence = - (1-a) * torch.log(lambda_p + 1e-6) + values
	return divergence.mean()


def closed_form_explore_reinforce(pn, acc, tol=1e-6):
	""" Computes the E-R halting probability regularizer through the formula:

		R(p_n, acc) = (1-acc)*(-H(pn)) + acc*E_pn[log(1+n)]

	but actually uses:

		R(p_n, acc) = E_pn[(1-acc)*log(pn) + acc*log(1+n)]

	as it is faster to compute.
	The use of E-R regularizer is motivated by the tradeoff between the Explore and Reinforce terms, mediated by the accuracy:
	- low accuracy incentivizes increasing the number of steps (Exploration/Trial-and-error)
	- high accuracy incentivizes reducing the number of steps (Reinforcement/Specialization)

	Parameters
	----------
	pn : Tensor
		tensor of halting probabilities of shape (B, N)
	acc : Tensor
		tensor of accuracies of shape (1,) or (B,), depending on the use of total accuracy or sample-wise accuracy.
		Sample-wise accuracy may give a more precise tradeoff which depends on the correctness of single examples:
		- examples with low accuracy will go towards increasing the number of steps
		- examples with high accuracy will go towards decreasing the number of steps  
	tol : float
		tolerance value to avoid log(0) cases
	
	Returns
	-------
	reg_loss : Tensor
		tensor of shape (1,) of the reduced regularization loss
	"""
	device = pn.device
	a = acc.unsqueeze(-1)
	N = pn.shape[-1]
	values = (1-a) * torch.log(pn+1e-6) + a * torch.log(torch.arange(1, N+1, dtype=torch.float, device=device))
	values = expected_value(pn, values)
	return values.mean()


class HaltingProbRegularizer:

	def __init__(self, reg_term, avg_steps=None):
		""" Regularizer term for halting probabilities

		Parameters
		----------
		reg_term : float
			multiplicative term for the regularizer (weight), also called beta
		avg_steps : float
			average number of steps, defined only for the geometric kl-div variant, where lambda_p = 1/avg_steps. If None, uses E-R regularization instead. Default None.
		"""

		# setup coefficients
		self.beta = reg_term
		self.vanilla = avg_steps is not None

		# setup lambda for log issues
		if self.vanilla:
			self.lambda_p = 1. / max(1., avg_steps)
			if self.lambda_p == 0.:
				self.lambda_p + 1e-3
			elif self.lambda_p == 1.:
				self.lambda_p - 1e-1
			self.loss = closed_form_kldiv_pondernet

		else:
			# setup loss function
			self.loss = closed_form_explore_reinforce

	def __call__(self, pn, acc=None, tol=1e-6):
		""" Computes one of the KL-div w/ geometric or the E-R halting probability regularizer, depending on whether avg_steps was set.

		Parameters
		----------
		pn : Tensor
			tensor of halting probabilities of shape (B, N)
		acc : Tensor
			tensor of accuracies of shape (1,) or (B,), depending on the use of total accuracy or sample-wise accuracy.
			Sample-wise accuracy may give a more precise tradeoff which depends on the correctness of single examples:
			- examples with low accuracy will go towards increasing the number of steps
			- examples with high accuracy will go towards decreasing the number of steps
			If using the KL-div regularizer, this argument is ignored.
			If None, and using E-R regularizer, an error may be raised. Default None.
		tol : float
			tolerance value to avoid log(0) cases
		
		Returns
		-------
		reg_loss : Tensor
			tensor of shape (1,) of the reduced regularization loss
		"""
		if self.vanilla:
			lambda_p = torch.tensor(self.lambda_p, device=pn.device)
			return self.beta * self.loss(pn, lambda_p)
		else:
			return self.beta * self.loss(pn, acc)
