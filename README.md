# tag-cat
## Time Adaptive, Grid - Convolutional Attention Transformer

## Introduction

The code in this repository refers to the paper in https://arxiv.org/abs/2207.02536 by Samuel Cognolato and Alberto Testolin. With this repository you will be able to try out the model defined in the paper, and customize it both in the model profiles or directly on the code. Further information will be given in the following sections.

Some functions or objects might be missing the documentation. It will surely be added in the future, but don't worry! You can still use the model profiles, and some test files are provided as a guide.

## Installation

The repository's code can be cloned with the command:

	git clone https://github.com/CognacS/tag-cat.git

or installed as packages with the command:

	pip install git+https://github.com/CognacS/tag-cat.git

## Model profiles
To make the job easier for those interested in checking out the model, but not getting their hands dirty, I've provided an interface to customize, store, and train/test at any time variants of the model. Model profiles are defined in the `json` format, which is very similar to python dictionaries. Model profiles define the structure and behavior of the model, and the arguments mimic the actual arguments used for defining the network modules. In `profiles/`, you will find three folders:
* `models/` includes the five variants (as **model profiles**) of the model shown in the paper https://arxiv.org/abs/2207.02536, that is the base model, the nogroup variant (performing additions with thousands of digits), the base_orig (using groups=heads in the attention mechanism), the base_vanillaponder (using the original KL-div for regularization of pondernet), and finally the fixed variant (not using dynamic halting).
* `batch_training/` contains a **batch profile** including all the 5 models above. This is particularly useful to train many variants with different seeds for many runs in one go.
* `datasets/` contains a **datasets profile** which defines the training and validation datasets used during the training procedure.
* `tests/` contains a **test profile**, useful to gather test results for many instances of the problem.

To run a training session on a single model, call the command:

	python main_train.py -m <model_profile> -d <datasets_profile> -o <output_path>

that is: train model `<model_profile>` on datasets `<datasets_profile>` and checkpoint at `<output_path>/<model_name>`.
This command will generate four files:
* `net_<model_name>.torch`, containing the model weights;
* `opt_<model_name>.torch`, containing the optimizer state;
* `sch_<model_name>.torch`, containing the LR-scheduler state;
* `log_<model_name>.pkl`, containing the training logs in pickle format.

To run a batched training session, that is, training different models for many runs with different seeds, run the command:

	python main_train.py -b <batch_profile> -d <datasets_profile> -o <output_path>

that is: train models included in `<batch_profile>`, which refers to many `<model_profile>`s, all on datasets `<datasets_profile>` and checkpoint all at `<output_path>/<model_name>` for all models.

To run a test session, call the command:

	python main_test.py -m <model_profile> -t <test_profile> -p <trained_models_path>

that is: test the model `<model_profile>` on the test suite `<test_profile>`, using the weights stored in `<trained_models_path>/<model_name>`. Notice that `<trained_models_path>` is exactly the same directory as `<output_path>` used during training.

To run a batched test session, that is, testing different models from many runs with different seeds, run the command:

	python main_test.py -b <batch_profile> -t <test_profile> -p <trained_models_path>

which will gather models the same as the `main_train.py` command.

Feel free to edit the models/procedures using the provided profiles, which should be enough to fully customize to your need.

If you need further information, call the help for these same commands using the `-h` flag.

## Editing the code
You are free to edit the architecture modules defined in the repository (that you can find in `tagcat/modules/`). The modular design should allow for localized changes. You can find sample code to define the dataset and the architecture in `tagcat/tests`.

## Conclusion
The repository is still in development. Feel free to contact me if you encounter any issue.