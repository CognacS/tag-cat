import argparse
from tagcat.factory import profile_reader, extract_batched_models, batch_train

def main():
	parser = argparse.ArgumentParser(description='Pipeline to train a network or a batch of networks')

	m_description = 'path to the model profile in JSON format. A model profile defines the arguments to construct a model, together with the training arguments.'
	parser.add_argument(
		'-m', metavar='<model_profile>', type=str, nargs=1,
		help = m_description
	)

	b_description = 'path to the batch profile in JSON format. A batch profile lists all models to train in batch, together with runs parameters, such as the #runs, etc.'
	parser.add_argument(
		'-b', metavar='<batch_profile>', type=str, nargs=1,
		help = b_description
	)

	d_description = 'path to the dataset profile in JSON format. A dataset profile can be composed as a list of datasets, each including the arguments to construct the datasets.'
	parser.add_argument(
		'-d', metavar='<dataset_profile>', type=str, nargs=1, required=True,
		help = d_description + ' In this setup it is mandatory to define a "train_dataset" and a "validation_dataset".'
	)

	p_description = 'output path where the trained network(s) directory(ies) will be created. If not defined, directories will be created in the same path as this file.'
	parser.add_argument(
		'-o', metavar='<out_path>', type=str, nargs=1,
		help = p_description
	)

	parser.add_argument('--v', action='store_true', help = 'set verbose for setup procedure.')
	parser.add_argument('--s', action='store_true', help = 'set silent for all procedure, i.e. no log will be generated.')
	parser.add_argument('--vt', action='store_true', help = 'set verbose for training procedure.')

	args = parser.parse_args()

	models_args = []
	if args.m is not None:
		model_profile = profile_reader(args.m[0])
		models_args.append(model_profile)

	if args.b is not None:
		batch_profile = profile_reader(args.b[0])
		models_args = extract_batched_models(batch_profile)

	dataset_profiles = None
	if args.d is not None:
		dataset_profiles = profile_reader(args.d[0])['datasets']

	if args.o is not None:
		output_path = args.o[0]
	else:
		output_path = None

	if output_path is None:
		output_path = '.'

	batch_train(
		models_args, dataset_profiles, output_path=output_path,
		verbose=args.v, silence=args.s, train_verbose=args.vt
	)



if __name__ == '__main__':
	main()