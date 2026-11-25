# building a model candidate
# retraining is not replacing the existing model
# it means creating a second model test it and decide if its better
# it should also pass check, until then original model is deployed

# retraining pipeline:
# Build retraining dataset: combine original training dataset with newly availble labelled referenced data
# Train candidate model: Train new Random Forest without touching deployed model
# Evaluate both models: Test incumbent and candidate on the same held out evaluation dataset
# candidate passes: eligible for promotion review
# Candidate fails: Rejecct candidate
# Promotion is a separate step. Neither outcome automatically changes the deployed model.

# An important distinction about the data
# 4 relevant datasets
# train.csv: Original training observations
# drift_referece.csv: Additional labelled observations for retraining
# evaluation.csv: Independent evaluation of the new candidate
# production.csv: Historical production replay used by monitoring and recovery decisions

# for first retraining experiment, combine train.csv and drift_reference.csv
# original model was trained only using part of the original training data
# there are now additional labelled observations available
# retraining gives the model an opportunity to learn from more examples

# the evaluation dataset was used in earlier model evaluation
# independent of candidate training

# retrain same model family and config initially
# Random Forest
# 200 trees
# class_weight="balanced_subsample"
# Random state 42
# Decision threshold 0.395

# If we simultaneously change the algorithm, 
# hyperparameters, threshold and training data,  
# won't know what caused any improvement or deterioration.

# compare evaluaton metrics and save the candidate model