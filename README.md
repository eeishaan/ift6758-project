# ift6758-project

Example commands to run personality model for regular training:
- cd ift6758-project-vc/scripts
- python train.py --input_path ../../new_data/Train/ --model personality_baseline --output_results_path ../trained_models

Example commands for evaluating submission of personality model on train/test split:
- python train.py --input_path ../../new_data/Train/ --model personality_baseline --eval_model True



