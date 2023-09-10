source ~/anaconda3/bin/activate pytorch1.13
echo "Starting feature extraction"
python feature_extraction.py /home/gts/projects/ecampbell/databases/
echo "Starting training and evaluation pipeline with interviewer in recordings"
python training_evaluation.py -i
echo "Starting training and evaluation pipeline without interviewer in recordings"
python training_evaluation.py