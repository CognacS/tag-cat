
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tasks.math.addition.generators import DEC_ADDITION_ALPHA, BASE_ADDITION_VOCABS, BatchedSumSeqDataset, get_expected_vocab_size
from tasks.utils.tokenizer import detokenizer_rich_chars2str, index_detokenize, vocab_reverse_map, index_tokenize, tokenizer_rich_str2chars

from .modules.seq2grid import GridSizeHandler
from .modules.pondernet import PonderNetModule
from .modules.architecture import SeqInputGridOutput, WrapperSIGO

from .functions.losses import get_class_weightings_pad, SampleWeightedLoss, HaltingProbRegularizer
from .checkpoints import CheckpointHandler

from .train import train_net
from .evaluation import test_net, compute_result

import json
from copy import deepcopy
from pathlib import Path


def profile_reader(profile_path):
	with open(profile_path) as f:
		profile = json.load(f)
	return profile


def extract_batched_models(batch_profile):
	model_profiles = []

	# for each model defined in the batch profile
	for model_data in batch_profile['models']:
		model_profile = profile_reader(model_data['path'])
		seed_start = model_data['seed_start']
		runs = model_data['runs']

		# for each run, each having a different seed copy the base arguments with the new seed
		for seed in range(seed_start, seed_start+runs):
			curr_model_profile = deepcopy(model_profile)
			# set name as <model_name>_<seed>
			curr_model_profile['model']['model_name'] = curr_model_profile['model']['model_name'] + f'_{seed}'
			# set seed
			curr_model_profile['training']['general']['random_seed'] = seed
			model_profiles.append(curr_model_profile)

	return model_profiles


def build_model(model_args, vocab_size, device=None):

	pondernet = None
	use_pondernet = 'architecture_components' in model_args
		
	if use_pondernet:
		arc_components = model_args['architecture_components']
		use_pondernet = 'pondernet' in model_args and arc_components['use_pondernet']
		

	if use_pondernet:
		pondernet = PonderNetModule(
			**model_args['general'],
			**model_args['module_components'],
			**model_args['pondernet'],
			device=device
		)

	body = SeqInputGridOutput(
		pondernet=pondernet,
		**model_args['general'],
		**model_args['module_components'],
		**model_args['attn_conv'],
		**model_args['s2g'],
		device=device
	)

	net = WrapperSIGO(body, vocab_size=vocab_size)
	net.to(device)

	return net, use_pondernet

def build_grid_size_handler(grid_size_args):
	return GridSizeHandler(**grid_size_args)

def build_loss(loss_args, pad_index, device=None):
	class_weights = get_class_weightings_pad(
		num_classes=len(DEC_ADDITION_ALPHA), pad_class=pad_index,
		pad_weight=0.1, device=device
	)
	loss_fn = SampleWeightedLoss(nn.CrossEntropyLoss(weight=class_weights))
	reg_fn = HaltingProbRegularizer(**loss_args)

	return loss_fn, reg_fn


def build_optimizer(optimizer_type, optimizer_args, net):
	no_wd_params = []
	wd_params = []
	no_wd = ['bias', 'embedding', 'rel_emb', 'q_emb', 'norm']

	for n, p in net.named_parameters():
		if any(s in n for s in no_wd):
			no_wd_params.append(p)
		else:
			wd_params.append(p)

	net_params = [
		{'params': no_wd_params, 'weight_decay': 0.0},
		{'params': wd_params, 'weight_decay': optimizer_args['weight_decay']}
	]

	if optimizer_type == 'adamw':
		return optim.AdamW(net_params, lr=optimizer_args['lr'])
	else:
		return None


def build_scheduler(scheduler_type, scheduler_args, optimizer, device=None):

	if scheduler_type == 'cosine':
		return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_args)
	else:
		return None


def profile_train(model_profile, train_dataset_args, validation_dataset_args, output_path=None, verbose=False, silence=False, train_verbose=False):
	# extract model arguments
	model_args = model_profile['model']
	train_args = model_profile['training']
	gen_train_args = train_args['general']

	# extract sub-arguments
	model_name = model_args['model_name']
	if not silence:
		header = f'|{" "*10}Starting training process for model: {model_name}{" "*10}|'
		print('-' * len(header))
		print(header)
		print('-' * len(header))

	# build directory
	if output_path is None:
		output_path = ''

	model_path = Path(f'{output_path}/{model_name}/')
	already_exists = model_path.exists()
	if not already_exists:
		model_path.mkdir(parents=True, exist_ok=True)

	# set seeds for reproducibility
	seed = train_args['general']['random_seed']
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if verbose:
		print(f' - random seed set to: {seed}')

	# get device
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	if verbose:
		print(f' - device set to: {device}')

	# setup datasets
	train_dataset = BatchedSumSeqDataset(**train_dataset_args)
	val_dataset = BatchedSumSeqDataset(**validation_dataset_args)
	vocab_size = train_dataset.get_vocab_size()
	if verbose:
		print(f' - training and validation datasets constructed')

	# setup grid size handler
	grid_size_handler = build_grid_size_handler(train_args['grid_size'])

	# setup model
	net, use_pondernet = build_model(model_args, vocab_size, device=device)
	if verbose:
		print(f' - network constructed')

	# setup losses
	loss_fn, reg_fn = build_loss(train_args['loss'], train_dataset.get_pad_index(), device=device)
	if verbose:
		print(f' - target loss constructed')
		print(f' - regularization loss constructed with beta={reg_fn.beta}, vanilla={reg_fn.vanilla}')

	# setup optimizer
	optimizer = build_optimizer(
		gen_train_args['optimizer'], train_args['optimizer'], net
	)
	if verbose:
		print(f' - using optimizer {train_args["general"]["optimizer"]}')

	# setup scheduler
	scheduler = build_scheduler(
		gen_train_args['scheduler'], train_args['scheduler'], optimizer
	)
	disable_scheduler = scheduler is None
	if verbose:
		print(f' - scheduler is enabled: {not disable_scheduler}')

	# setup other params
	rec_args = train_args['recurrence_params']
	train_max_steps, val_max_steps = rec_args['train_max_steps'], rec_args['valid_max_steps']
	grad_clipping = gen_train_args['grad_clipping']
	epochs = gen_train_args['epochs']
	checkpoint_period = gen_train_args['checkpoint_period']

	# setup data gathering
	loss_logs = {'train_loss': [], 'tr_acc': [], 'val_acc':[], 'epoch': 0}
	if use_pondernet:
		loss_logs['n_updates_train'] = {'min': [], 'max': [], 'mean':[], 'std':[]}
		loss_logs['n_updates_eval'] = {'min': [], 'max': [], 'mean':[], 'std':[]}

	# setup checkpoint handler
	checkpoint_handler = CheckpointHandler(
		base_filename=model_name, path=str(model_path)+'/',
		checkpoint_period=checkpoint_period,
		network=net, optimizer=optimizer, scheduler=scheduler,
		logs=loss_logs, log_epoch=True, device=device
	)

	# load state if it already exists
	if already_exists:
		checkpoint_handler.load_structures()

	if verbose:
		print(f'Training from {loss_logs["epoch"]} to {epochs} epochs')

	# train net
	train_net(
		net, loss_fn=loss_fn, reg_fn=reg_fn,
		optimizer=optimizer, scheduler=scheduler,
		epochs=epochs, grid_size_handler=grid_size_handler,
		train_max_steps=train_max_steps, train_dataset=train_dataset,
		use_pondernet=use_pondernet, grad_clipping=grad_clipping,
		disable_scheduler=disable_scheduler,
		val_max_steps=val_max_steps, val_dataset=val_dataset,
		checkpoint_handler=checkpoint_handler,
		device=device, loss_logs=loss_logs, silence=silence, verbose=train_verbose
	)


def profile_test(model_profile, test_profile, models_path=None, verbose=False, silence=False, test_verbose=False):
	# extract model arguments
	model_args = model_profile['model']
	train_args = model_profile['training']
	max_steps = train_args['recurrence_params']['valid_max_steps']
	general_args = test_profile['general']
	operands_nums = test_profile['operands_num']
	operands_sizes = test_profile['operands_size']
	random_seed = test_profile['options']['random_seed']

	# extract sub-arguments
	model_name = model_args['model_name']
	if not silence:
		header = f'|{" "*10}Starting testing process for model: {model_name}{" "*10}|'
		print('-' * len(header))
		print(header)
		print('-' * len(header))

	# build directory
	if models_path is None:
		models_path = ''

	model_path = Path(f'{models_path}/{model_name}/')
	if not model_path.exists():
		return

	# get device
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	if verbose:
		print(f' - device set to: {device}')

	# extract expected vocabulary size
	numeric_base = 10
	if 'numeric_base' in test_profile['general']:
		numeric_base = test_profile['general']['numeric_base']
	vocab_size = get_expected_vocab_size(numeric_base)

	# setup model
	net, use_pondernet = build_model(model_args, vocab_size, device=device)
	if verbose:
		print(f' - network constructed')

	# setup grid size handler
	grid_size_handler = build_grid_size_handler(train_args['grid_size'])

	# setup checkpoint handler and load model
	checkpoint_handler = CheckpointHandler(
		base_filename=model_name, path=str(model_path)+'/',
		network=net, device=device
	)
	checkpoint_handler.load_structures()

	# start extracting performances
	perf_per_dataset = {}

	for operands_num, operands_size in zip(operands_nums, operands_sizes):

		dataset_name = f'[{operands_num},{operands_size}]'
			
		dataset_args = {'operands_num': operands_num, 'operands_size': operands_size, **general_args}
		test_dataset = BatchedSumSeqDataset(**dataset_args)

		# setup seed
		random.seed(random_seed)
		np.random.seed(random_seed)
		torch.manual_seed(random_seed)

		# test network on dataset
		if verbose: print(f'Testing {model_name} on {dataset_name}')
		perf = test_net(net, max_steps=max_steps, test_dataset=test_dataset, grid_size_handler=grid_size_handler, device=device, verbose=test_verbose)
		perf_per_dataset[dataset_name] = perf

	return model_name, perf_per_dataset

def oneop_test(model_profile, operation, num_lists, list_size, max_steps, models_path=None):
	# extract model arguments
	model_args = model_profile['model']
	train_args = model_profile['training']
	max_steps = train_args['recurrence_params']['valid_max_steps']

	# extract sub-arguments
	model_name = model_args['model_name']

	# build directory
	if models_path is None:
		models_path = ''

	model_path = Path(f'{models_path}/{model_name}/')
	if not model_path.exists():
		return

	# get device
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	# extract expected vocabulary size
	vocab_size = get_expected_vocab_size(10)

	# setup model
	net, use_pondernet = build_model(model_args, vocab_size, device=device)

	# setup checkpoint handler and load model
	checkpoint_handler = CheckpointHandler(
		base_filename=model_name, path=str(model_path)+'/',
		network=net, device=device
	)
	checkpoint_handler.load_structures()

	vocab = BASE_ADDITION_VOCABS[10]
	tokens = index_tokenize(operation, vocab, tokenizer_rich_str2chars)

	result, n_updates = compute_result(net, tokens, num_lists, list_size, max_steps, device=device)

	result = index_detokenize(result, vocab_reverse_map(vocab), detokenizer_rich_chars2str)

	return result, n_updates


def batch_train(model_profiles, dataset_profiles, output_path=None, verbose=False, silence=False, train_verbose=False):
	tr_dataset = dataset_profiles['train_dataset']
	val_dataset = dataset_profiles['validation_dataset']
	for model_profile in model_profiles:
		profile_train(model_profile, tr_dataset, val_dataset, output_path, verbose, silence, train_verbose)

def batch_test(model_profiles, test_profile, models_path=None, verbose=False, silence=False, test_verbose=False):
	performance_table = {}
	for model_profile in model_profiles:
		model_name, perf_per_dataset = profile_test(model_profile, test_profile, models_path, verbose, silence, test_verbose)
		performance_table[model_name] = perf_per_dataset
	return performance_table