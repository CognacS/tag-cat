from setuptools import setup, find_packages

setup(
	name='tagcat',
	version='1.0',
	description='tag-cat model for solving mathematic problems',
	author='Samuel Cognolato',
	author_email='samuel.cognolato@studenti.unipd.it',
	url='https://github.com/CognacS/tag-cat',
	install_requires=[
        'numpy>=1.21.3',
		'pandas>=1.3.0',
		'torch>=1.11.0'
    ],
	packages=find_packages(include=['tagcat', 'tasks', 'tagcat.*', 'tasks.*'])
)