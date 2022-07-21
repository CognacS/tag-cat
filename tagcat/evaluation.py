import torch

from .functions.metrics import compute_char_acc, compute_seq_acc

def test_net(net, max_steps, test_dataset, grid_size_handler, device=None, verbose=False):

	# to evaluation mode
	net.eval()

	all_acc_char_level = []
	all_acc_seq_level = []
	all_n_updates = []

	with torch.no_grad():
		for i in range(len(test_dataset)):

			batch, = test_dataset[i]

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
			
			# pad target to list_size
			y_batch = torch.functional.F.pad(
				y_batch, pad=(list_size-y_batch.shape[-1], 0), value=test_dataset.get_pad_index()
			)

			################### 3 - FORWARD PASS ######################
			out, n_updates = net(x_batch, num_lists, list_size, max_steps)

			all_predictions = torch.argmax(out, dim=2)
			all_labels = y_batch

			# compute char/seq accuracies
			test_accuracy_char_level = compute_char_acc(all_predictions, all_labels, compute_argmax=False,
												ignore_symb=test_dataset.get_pad_index()).cpu().item()
			test_accuracy_seq_level = compute_seq_acc(all_predictions, all_labels, compute_argmax=False,
												ignore_symb=test_dataset.get_pad_index()).cpu().item()

			# save outputs and labels
			all_acc_char_level.append(test_accuracy_char_level)
			all_acc_seq_level.append(test_accuracy_seq_level)
			all_n_updates.append(n_updates)


	# concatenate all the outputs and labels in a single tensor
	all_n_updates = torch.cat(all_n_updates)

	test_accuracy_char_level = sum(all_acc_char_level) / len(all_acc_char_level)
	test_accuracy_seq_level = sum(all_acc_seq_level) / len(all_acc_seq_level)

	if verbose:
		print(f'pondering steps min/max/avg: {all_n_updates.min()}/{all_n_updates.max()}/{all_n_updates.mean()}')
		print(f'Test char/seq accuracy: {test_accuracy_char_level}/{test_accuracy_seq_level}')

	return {'char': test_accuracy_char_level, 'seq': test_accuracy_seq_level}
