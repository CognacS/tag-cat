import torch
import pickle


def craft_filename(pondernet, highway, reset, norm_pos):
  rec = 'ponder' if pondernet else 'fixrec' # pondernet or fixed
  hway = 'hiway' if highway else 'resid'  # highway or residual
  rset = 'reset' if reset else 'norst'    # reset or pass through
  norm = norm_pos + 'norm' # post/pre/none
  return '_'.join([rec, hway, rset, norm])


def save_network(filename, net):
	net_state_dict = net.state_dict()
	torch.save(net_state_dict, filename)
	print("Network saved")

def save_optimizer(filename, optimizer):
	opt_state_dict = optimizer.state_dict()
	torch.save(opt_state_dict, filename)
	print("Optimizer saved")

def save_scheduler(filename, scheduler):
	sch_state_dict = scheduler.state_dict()
	torch.save(sch_state_dict, filename)
	print("Scheduler saved")

def save_logs(filename, logs):
	a_file = open(filename, "wb")
	pickle.dump(logs, a_file)
	a_file.close()
	print('Logs saved')


def load_network(filename, net, device=None):
	net_state_dict = torch.load(filename, map_location=device)
	# Update the network parameters
	net.load_state_dict(net_state_dict)
	net.to(device)
	print('Network loaded')


def load_optimizer(filename, optimizer, device=None):
	opt_state_dict = torch.load(filename, map_location=device)
	optimizer.load_state_dict(opt_state_dict)
	print('Optimizer loaded')

def load_scheduler(filename, scheduler, device=None):
	sch_state_dict = torch.load(filename, map_location=device)
	scheduler.load_state_dict(sch_state_dict)
	print('Scheduler loaded')


def load_logs(filename, logs, device=None):
	a_file = open(filename, "rb")
	loss_log = pickle.load(a_file)
	for k in loss_log:
		logs[k] = loss_log[k]
	print('Logs loaded')


NET_KEY = 'net'
OPT_KEY = 'opt'
SCH_KEY = 'sch'
LOG_KEY = 'log'

PREF_SUFF = {
	NET_KEY : (NET_KEY + '_', '.torch'),
	OPT_KEY : (OPT_KEY + '_', '.torch'),
	SCH_KEY : (SCH_KEY + '_', '.torch'),
	LOG_KEY : (LOG_KEY + '_', '.pkl')
}

SAVE_FUNCTS = {
	NET_KEY : save_network,
	OPT_KEY : save_optimizer,
	SCH_KEY : save_scheduler,
	LOG_KEY : save_logs
}

LOAD_FUNCTS = {
	NET_KEY : load_network,
	OPT_KEY : load_optimizer,
	SCH_KEY : load_scheduler,
	LOG_KEY : load_logs
}

def save_structures(base_filename, dict_struct, additional_path=None):

	# for each structure+load function
	for k, save in SAVE_FUNCTS.items():
		curr_struct = dict_struct[k]

		# save the structure if it exists
		if curr_struct is not None:
			pref, suff = PREF_SUFF[k]
			curr_filename = pref + base_filename + suff
			if additional_path is not None:
				curr_filename = additional_path + curr_filename
			save(curr_filename, curr_struct)

def load_structures(base_filename, dict_struct, additional_path=None, device=None):

	# for each structure+load function
	for k, load in LOAD_FUNCTS.items():
		curr_struct = dict_struct[k]

		# load the structure if it exists
		if curr_struct is not None:
			pref, suff = PREF_SUFF[k]
			curr_filename = pref + base_filename + suff
			if additional_path is not None:
				curr_filename = additional_path + curr_filename
			load(curr_filename, curr_struct, device)


class CheckpointHandler:

	def __init__(self, base_filename, path=None, checkpoint_period=0., network=None, optimizer=None, scheduler=None, logs=None, log_epoch=False, device=None):
		
		self.base_filename = base_filename
		self.path = path
		self.checkpoint_period = checkpoint_period
		self.dict_struct = {
			NET_KEY : network,
			OPT_KEY : optimizer,
			SCH_KEY : scheduler,
			LOG_KEY : logs
		}
		self.log_epoch = log_epoch
		self.device = device

		self.epoch = None
		self.max_epochs = None


	def load_structures(self, filename=None):
		# if no filename is provided, use the one given in the costructor
		if filename is None:
			filename = self.base_filename

		load_structures(filename, self.dict_struct, additional_path=self.path, device=self.device)


	def save_structures(self):
		save_structures(self.base_filename, self.dict_struct, additional_path=self.path)


	def update_epoch(self, epoch):
		self.epoch = epoch
		if self.log_epoch:
			self.dict_struct[LOG_KEY]['epoch'] = self.epoch


	def set_max_epochs(self, max_epochs):
		self.max_epochs = max_epochs


	def __iter__(self):
		assert self.max_epochs is not None, 'max_epochs has not been set. Call set_max_epochs before iterating'

		if self.log_epoch and 'epoch' in self.dict_struct[LOG_KEY] and self.dict_struct[LOG_KEY] is not None:
			epoch = self.dict_struct[LOG_KEY]['epoch']
		else:
			epoch = 0

		self.update_epoch(epoch)
		return self

	def __next__(self):
		if self.epoch % self.checkpoint_period == 0:
			self.save_structures()

		if self.epoch >= self.max_epochs:
			raise StopIteration

		curr_epoch = self.epoch
		self.update_epoch(self.epoch+1)

		return curr_epoch
