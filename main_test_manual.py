import argparse
import pandas as pd

from tagcat.factory import profile_reader, oneop_test

def main():
	parser = argparse.ArgumentParser(description='Pipeline to test a network or a batch of networks')

	m_description = 'path to the model profile in JSON format. A model profile defines the arguments to construct a model, together with the training arguments.'
	parser.add_argument(
		'-m', metavar='<model_profile>', type=str, nargs=1,
		help = m_description
	)

	p_description = 'models path where the trained network(s) directory(ies) are stored.'
	parser.add_argument(
		'-p', metavar='<models_path>', type=str, nargs=1,
		help = p_description
	)

	args = parser.parse_args()

	model_profile = None
	if args.m is not None:
		model_profile = profile_reader(args.m[0])

	models_path = args.p[0]

	operation = input('Operation: ')
	num_lists = int(input('Rows: '))
	list_size = int(input('Cols: '))
	max_steps = int(input('Max steps: '))


	result, n_updates = oneop_test(
		model_profile, operation, num_lists, list_size, max_steps, models_path
	)

	correct_result = 0
	for num in operation[:-1].split('+'):
		correct_result += int(num)

	print('Number updates: ', n_updates)
	print('Predicted result: ', result)
	print('Correct result: ', correct_result)



if __name__ == '__main__':
	main()