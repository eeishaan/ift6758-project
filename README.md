# IFT6758 Project - Autumn 2019
## Team 17

Authors:
Philippe Lelièvre,
Julien Horwood,
Ishaan Kumar,
Vicki Anand

Project Organization
------------
    ├── dummy_data        <- Fake data for testing purpose
    ├── mappings          <- Baseline model mapping (used for the baseline submission)
    ├── models            <- Models used for the experiments
    ├── notebooks         <- Notebooks used mainly for data visualization and prototyping
    ├── scripts           <- All training and experiments scripts
    ├── test_results      <- Examples of test prediction results
    ├── trained_model     <- Trained models folder
    ├── utils             <- Utilities used accross the application
    ├── xgboost           <- [IGNORE THIS FOLDER FOR THE CODE EVALUATION] Source code of the XGBoost project (https://github.com/dmlc/xgboost) to work around the problem that we cannot install a library on the environment. To be removed in the future if we can install the library.
    ├── ift6758           <- Evaluation script
    └── README.md         <- The top-level README for developers using this project.
--------

## Usage:

### Train the model
- python scripts/train.py --input_path ../../new_data/Train/ --model final --output_results_path ../trained_models

### Train and evaluate the model on train/test split
- python scripts/train.py --input_path ../../new_data/Train/ --model final --eval_model True

### Perform k-fold cross-validation on the model 
- python scripts/train.py --input_path ../../new_data/Train/ --model final --k_fold True

## Writing Conventions
* Limit all lines to a maximum of 119 characters. Only exception is when it impact negatively the readibility of a sentence.

## xgboost folder
Like mentioned in the project structure section, we included the source code of the XGBoost project (https://github.com/dmlc/xgboost) to work around the problem that we cannot install any library on the main environment.

For the evaluation procedure, ignore the code inside the folder named xgboost

This folder should be removed in the future if we can install the library on the main python environment.
