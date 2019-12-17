import argparse
import os
import sys

sys.path.append('../')  # TODO fix these imports properly

from models.age_estimator import AgeEstimator
from models.baselines import MeanRegressor, MajorityClassifier
from utils.k_fold import k_fold

from models.final_estimator import SingleTaskEstimator
from models.gender_estimator import TreeEnsembleEstimator
from models.personality_estimators import PersonalityTreeRegressor
from utils.data_processing import parse_input, split_data

# TODO: Make a model selection script
# TODO: Implement custom transformers for image, text, relational data (i.e. modularize notebook code

gender_clf = TreeEnsembleEstimator()
ope_reg = PersonalityTreeRegressor(150, 30)
con_reg = PersonalityTreeRegressor(20, None)
ext_reg = PersonalityTreeRegressor(25, None)
agr_reg = PersonalityTreeRegressor(120, 30)
neu_reg = PersonalityTreeRegressor(10, 30)
age_clf = AgeEstimator(gender_clf=gender_clf, ope_reg=ope_reg, con_reg=con_reg, ext_reg=ext_reg, agr_reg=agr_reg,
                       neu_reg=neu_reg)

MODEL_MAPPING = {
    'final': SingleTaskEstimator(
        age_clf=age_clf,
        gender_clf=gender_clf,
        ope_reg=ope_reg,
        con_reg=con_reg,
        ext_reg=ext_reg,
        agr_reg=agr_reg,
        neu_reg=neu_reg
    ),
    'baseline': SingleTaskEstimator(
        age_clf=MajorityClassifier(),
        gender_clf=MajorityClassifier(),
        ope_reg=MeanRegressor(),
        con_reg=MeanRegressor(),
        ext_reg=MeanRegressor(),
        agr_reg=MeanRegressor(),
        neu_reg=MeanRegressor()
    ),
    'gender_only': SingleTaskEstimator(
        age_clf=MajorityClassifier(),
        gender_clf=TreeEnsembleEstimator(),
        ope_reg=MeanRegressor(),
        con_reg=MeanRegressor(),
        ext_reg=MeanRegressor(),
        agr_reg=MeanRegressor(),
        neu_reg=MeanRegressor()
    ),
    'personality_baseline': SingleTaskEstimator(
        age_clf=MajorityClassifier(),
        gender_clf=TreeEnsembleEstimator(),
        ope_reg=PersonalityTreeRegressor(150, 30),
        con_reg=PersonalityTreeRegressor(20, None),
        ext_reg=PersonalityTreeRegressor(25, None),
        agr_reg=PersonalityTreeRegressor(120, 30),
        neu_reg=PersonalityTreeRegressor(10, 30)
    ),

}


def train(input_path, output_path, model_name, model_eval, k_fold_mode):
    os.makedirs(output_path, exist_ok=True)
    age_to_group = True
    X, y = parse_input(input_path, age_to_group=age_to_group)
    model = MODEL_MAPPING[model_name]

    if k_fold_mode:
        k_fold(X, y, model, age_to_group, n_splits=5)
        return
    if model_eval:
        Xtrain, Xtest, ytrain, ytest = split_data(X, y)
        model.fit(Xtrain, ytrain)
        model.eval(Xtest, ytest, age_to_group=age_to_group)
    else:
        model.fit(X, y)
        model.save(os.path.join(output_path, model_name + '.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None,
                        help='Input path')
    parser.add_argument('--output_results_path', type=str, default='../trained_models',
                        help='Output dir for trained model')
    parser.add_argument('--model', type=str, default='baseline',
                        help='Specify which model to train')
    parser.add_argument('--model_eval', type=bool, default=False,
                        help='Whether or not evaluate model on train/test split. False by default. Model will not be saved if set.')
    parser.add_argument('--k_fold', type=bool, default=False,
                        help='Run k-folding instead of training')
    args = parser.parse_args()
    train(args.input_path, args.output_results_path, args.model, args.model_eval, args.k_fold)
