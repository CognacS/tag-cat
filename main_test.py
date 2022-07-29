import argparse
import pandas as pd

from tagcat.factory import profile_reader, extract_batched_models, batch_test

def main():
	parser = argparse.ArgumentParser(description='Pipeline to test a network or a batch of networks')

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

	t_description = 'path to the test profile in JSON format. A test profile is composed of common dataset arguments, and lists of varying arguments for many test beds.'
	parser.add_argument(
		'-t', metavar='<test_profile>', type=str, nargs=1,
		help = t_description
	)

	p_description = 'models path where the trained network(s) directory(ies) are stored.'
	parser.add_argument(
		'-p', metavar='<models_path>', type=str, nargs=1,
		help = p_description
	)

	parser.add_argument('--v', action='store_true', help = 'set verbose for setup procedure.')
	parser.add_argument('--s', action='store_true', help = 'set silent for all procedure, i.e. no log will be generated.')
	parser.add_argument('--vt', action='store_true', help = 'set verbose for test procedure.')

	args = parser.parse_args()

	models_args = []
	if args.m is not None:
		model_profile = profile_reader(args.m[0])
		models_args.append(model_profile)

	if args.b is not None:
		batch_profile = profile_reader(args.b[0])
		models_args = extract_batched_models(batch_profile)

	test_profiles = None
	if args.t is not None:
		test_profiles = profile_reader(args.t[0])

	models_path = args.p[0]

	performances_table = batch_test(
		models_args, test_profiles, models_path=models_path,
		verbose=args.v, silence=args.s, test_verbose=args.vt
	)

	df = pd.DataFrame(performances_table).transpose()

	pd.set_option('display.max_rows', None)
	pd.set_option('display.max_columns', None)
	print(df)



if __name__ == '__main__':
	main()