# MLFLOW experiment tracking and model registry stage
# system does not have a proper, queryable history of every training run
# if 20 candidates were run over the course of several weeks, one should be able to say:
# which training dataset was used by each, hyperparameters used, metrics did each run acheive. why was a candidate rejected, model version approved for deployement