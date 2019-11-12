import argparse
import os
import sys

sys.path.append('../')  # TODO fix these imports properly
from models.baselines import MajorityClassifier, MeanRegressor
from models.final_estimator import SingleTaskEstimator
from models.gender_estimator import TreeEnsembleEstimator
from models.personality_estimators import PersonalityTreeRegressor
from utils.data_processing import parse_input



MODEL_MAPPING = {
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
        ope_reg=PersonalityTreeRegressor(),
        con_reg=PersonalityTreeRegressor(),
        ext_reg=PersonalityTreeRegressor(),
        agr_reg=PersonalityTreeRegressor(),
        neu_reg=PersonalityTreeRegressor()
    )
}


def train(input_path, output_path, model_name):
    os.makedirs(output_path, exist_ok=True)
    X, y = parse_input(input_path)
    model = MODEL_MAPPING[model_name]

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
    args = parser.parse_args()
    train(args.input_path, args.output_results_path, args.model)
