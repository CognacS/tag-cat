from distutils.core import setup, find_packages

setup(
	name='tagcat',
	version='1.0',
	description='tag-cat model for solving mathematic problems',
	author='Samuel Cognolato',
	author_email='samuel.cognolato@studenti.unipd.it',
	url='https://github.com/CognacS/tag-cat',
	packages=find_packages(include=['tagcat.*', 'tasks.*']),
)