source ~/anaconda3/bin/activate pytorch1.13
echo "Starting feature extraction"
python feature_extraction.py
echo "Starting training and evaluation pipeline"
python training_evaluation.py
