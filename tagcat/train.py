import time
import numpy as np
import torch

from .functions.metrics import compute_char_acc_samplewise, compute_char_acc, compute_seq_acc
from .modules.pondernet import reconstruct_probabilities

STATS_PROFILE = {'functions':[torch.min, torch.max, torch.mean, torch.std], 'st_names':['min', 'max', 'mean', 'std']}

def add_statistics_to_dict(src_array, dst_dict=None, functions=STATS_PROFILE['functions'], st_names=STATS_PROFILE['st_names']):

  stats = {}
  for fun, name in zip(functions, st_names):
    stat = fun(src_array).cpu().item()
    stats[name] = stat
    if dst_dict is not None:
      dst_dict[name].append(stat)

  return stats


def train_net(
		net, loss_fn, reg_fn,
		optimizer, scheduler, epochs, grid_size_handler,
		train_max_steps, train_dataset, use_pondernet,
		grad_clipping=10., disable_scheduler=False,
		val_max_steps=None, val_dataset=None, checkpoint_handler=None,
		device=None, loss_logs=None, verbose=False, silence=False
	):


	##################################### INITIALIZATION #####################################
	# create training_data for the halting distribution if using pondernet
	training_data = None
	if use_pondernet:
		training_data = {
			'decoder': net.decoder, 'target': None, 'loss_fn': loss_fn,
			'loss_value': None, 'running_indices': None, 'h_probs': None
		}

	# create epochs iterator, including the possibility of using a checkpoint handler which will save the state periodically
	if checkpoint_handler is None:
		epochs_iter = range(epochs)
	else:
		epochs_iter = checkpoint_handler
		epochs_iter.set_max_epochs(epochs)

	##################################### TRAINING LOOP #####################################
	for epoch in epochs_iter:	

		# start EPOCH timer
		epoch_start_time = time.time()

		if not silence:
			if use_pondernet:
				header = '| lr      | val update# m/M/E | tr update# m/M/E  | CEloss     | reg_loss   | tot_loss   | tr_acc  | val_acc |'
			else:
				header = '| lr      | CEloss     | tr_acc  | val_acc |'
			
			print('-' * len(header))
			print(header)
			print('-' * len(header))

		# start loggers
		train_loss = []
		fun_loss = []
		reg_loss = []

		# loop on super batches, which contains mini-batches (based on dividing instances on their size)
		# using range(len(...)) to avoid an infinite loop, because the iterator on train_dataset does not
		# have an end
		for i in range(len(train_dataset)):

			# start BATCH timer
			start_time = time.time()

			################################# TRAINING PHASE #################################
			# set network training mode
			net.train()

			# get super-batch
			super_batch = train_dataset[i]

			# startup target and regularization losses (accumulated over the super-batch)
			f_loss = torch.tensor(0., device=device) # target loss
			r_loss = torch.tensor(0., device=device) # regular. loss
			f_acc = 0. # target accuracy
			
			n_updates_eval = []
			n_updates_train = []

			# for each batch inside the super batche
			for batch in super_batch:

				################### 1 - DATA EXTRACTION ###################
				# extract data from the batch
				x_batch, y_batch = batch['seq'], batch['res'], 
				max_opsnum, max_opsize = batch['max_opsnum'], batch['max_opsize']

				################### 2 - DATA SETUP #######################
				# move data to device
				x_batch = x_batch.to(device)
				y_batch = y_batch.long().to(device)

				# decide grid sizes
				num_lists, list_size = grid_size_handler.decide_grid_size(
					max_opsnum=max_opsnum, max_opsize=max_opsize
				)
				if verbose and not silence:
					print(f'num_lists(rows) = {num_lists}, list_size(cols) = {list_size}')

				# pad target to list_size
				y_batch = torch.functional.F.pad(
					y_batch, pad=(list_size-y_batch.shape[-1], 0), value=train_dataset.get_pad_index()
				)

				# setup inside-net training data
				if use_pondernet:
					training_data['target'] = y_batch
					training_data['loss_value'] = torch.tensor(0., device=device)
					training_data['running_indices'] = []
					training_data['h_probs'] = []

				################### 3 - FORWARD PASS ######################
				out, curr_n_updates = net(x_batch, num_lists, list_size, train_max_steps, training_data=training_data)


				################### 4 - TARGET LOSS ACCUMULATION ##########
				# compute main loss
				if use_pondernet:
					# get accumulated loss (expected value)
					curr_f_loss = training_data['loss_value']
				else:
					# compute loss now
					curr_f_loss = loss_fn(out, y_batch)
				f_loss = f_loss + curr_f_loss

				################### 5 - COMPUTE ACCURACY ##################
				# compute training accuracy on the mini-batch
				with torch.no_grad():
					net.eval() # eval mode
					# recompute result in evaluation mode (different!!)
					out_tr, curr_n_updates_eval = net(x_batch, num_lists, list_size, train_max_steps)
					curr_f_acc_samples = compute_char_acc_samplewise(
						output=out_tr, target=y_batch, ignore_symb=train_dataset.get_pad_index()
					)
					net.train() # back to train mode

				curr_f_acc = curr_f_acc_samples.mean().cpu().item()
				if verbose and not silence:
					print('Mini-batch accuracy:', curr_f_acc)
				f_acc += curr_f_acc

				################### 6 - REGULARIZ. LOSS ACCUMULATION ########
				# compute regularization loss for pondernet
				if use_pondernet:
					# recostruct the halting probabilities for all steps (also those missing)
					probs = reconstruct_probabilities(
						training_data['running_indices'], training_data['h_probs'])
					if verbose and not silence:
						print(probs[0].detach().cpu().numpy())
					curr_r_loss = reg_fn(probs, curr_f_acc_samples)
					r_loss = r_loss + curr_r_loss
					# log n_updates
					n_updates_eval.append(curr_n_updates_eval.detach())
					n_updates_train.append(curr_n_updates.detach())

			# accumulate losses for each mini-batch in a super-batch
			f_loss = f_loss / len(super_batch)
			f_acc =  f_acc  / len(super_batch)
			if use_pondernet:
				r_loss = r_loss / len(super_batch)
				n_updates_eval = torch.cat(n_updates_eval)
				n_updates_train = torch.cat(n_updates_train)
			# accumulate total loss
			t_loss = f_loss + r_loss

			# backpropagation on the super-batch
			net.zero_grad()
			t_loss.backward()

			# gradient clipping
			torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clipping)

			# GRADIENT CHECK
			if verbose and not silence:
				total_norm = []
				for n, p in net.named_parameters():
					if p.grad is not None:
						param_norm = p.grad.detach().data.norm(2)
						total_norm.append(param_norm.cpu().item())
						if param_norm.cpu().item() > 100.:
							print(
								f'{n} over the threshold: {param_norm.cpu().item()}')
				total_norm = np.array(total_norm)
				print(
					f'TOTAL NORM min/max/avg: {np.min(total_norm)}/{np.max(total_norm)}/{np.mean(total_norm)}')
			# GRADIENT CHECK

			# Update the weights
			optimizer.step()

			################################# VALIDATION PHASE #################################
			val_acc = 0.0 # initialize validation accuracy
			if val_dataset is not None:
				val_batch = val_dataset[0][0] # generate an instance (index does not matter)
				# handle data
				x_batch, y_batch = val_batch['seq'], val_batch['res']
				max_opsnum, max_opsize = val_batch['max_opsnum'], val_batch['max_opsize']
				x_batch = x_batch.to(device)
				y_batch = y_batch.long().to(device)
				num_lists, list_size = grid_size_handler.decide_grid_size(
					max_opsnum=max_opsnum, max_opsize=max_opsize
				)

				# start validation
				net.eval()
				with torch.no_grad():
					# pad target to list_size
					y_batch = torch.functional.F.pad(y_batch, pad=(
						list_size-y_batch.shape[1], 0), value=val_dataset.get_pad_index())
					# forward pass
					out, n_updates_val = net(x_batch, num_lists, list_size, val_max_steps)
					val_acc = compute_char_acc(
						out, y_batch, ignore_symb=val_dataset.get_pad_index()
					).cpu().item()

			# save train loss for this batch
			f_loss_batch = f_loss.detach().cpu().item()
			fun_loss.append(f_loss_batch)
			r_loss_batch = r_loss.detach().cpu().item()
			reg_loss.append(r_loss_batch)
			t_loss_batch = t_loss.detach().cpu().item()

			# log losses
			loss_logs['train_loss'].append(f_loss_batch)
			loss_logs['tr_acc'].append(f_acc)
			loss_logs['val_acc'].append(val_acc)

			# compute elapsed time from the start of the epoch
			elapsed_time = time.time() - start_time

			################################# LOGGING PHASE #################################
			if use_pondernet:
				# compute all statistics for evaluation and training n_updates logged data
				eval_stats = add_statistics_to_dict(
					src_array=n_updates_eval, dst_dict=loss_logs['n_updates_eval'])
				train_stats = add_statistics_to_dict(
					src_array=n_updates_train, dst_dict=loss_logs['n_updates_train'])

				if not silence:
					# print log string
					print(
						'| {:01.1e} | {:5.2f}/{:5.2f}/{:5.2f} | {:5.2f}/{:5.2f}/{:5.2f} | '
						'{:+02.3e} | {:+02.3e} | {:+02.3e} | {:6.3f}  | {:6.3f}  |'.format(
							scheduler.get_last_lr()[0],
							eval_stats['min'], eval_stats['max'], eval_stats['mean'],
							train_stats['min'], train_stats['max'], train_stats['mean'],
							f_loss_batch, r_loss_batch, t_loss_batch, f_acc, val_acc
						)
					)
			else:
				if not silence:
					# print log string
					print(
						'| {:01.1e} | {:+02.3e} | {:6.3f}  | {:6.3f}  |'.format(
							scheduler.get_last_lr()[0],
							f_loss_batch, f_acc, val_acc
						)
					)

		# save average train loss
		train_loss = np.mean(fun_loss)

		if not disable_scheduler:
			scheduler.step()

		if not silence:
			bottom = '| end of epoch {:3d} | time: {:5.2f}s | epoch total loss {:02.3e} | '.format(
				epoch, (time.time() - epoch_start_time), train_loss
			)
			print('-' * len(bottom))
			print(bottom)
			print('-' * len(bottom))