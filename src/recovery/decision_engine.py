# when data cchanges, how does the platform decide whether to leave the model, investigate or initiate recovery
# system detects drift n 29 features, doesnt know whether the model is actually performing worse

# 2 things to monitor
# Drift Drift
# if inputs eaching the model have changed
# transaction amounts in production may look differently from transaction amounts in refernce dataset

# model performance degradation
# is the model making worse predictions
# for feaud detection:
# precision: of all the transactions flagged, how many were actually fraud
# recall: of all the fraud cases, how many did the model catch
# F1 Score: A combined measure of precision and recall
# PR AUC: how well the model distinguishes fraud from legit transacions across different decision thresholds

# A model can experience input drift while maintaining good performance. 
# # It can also experience performance degradation without every feature showing significant drift.

# Why labels matter:
# model makes predictions before knowing the actual outcome     ss       
# if a transaction arrives, model predicts fraud, API records prediction, several days later the oucome bcomes available
# till then one cannot reliably calculate if the prediction was correct 

# Decisions taken by Engine:
# No drift and perf healthy     CONTINUE_MONITORING
# Drift detected, performance healthy       INVESTIGATE_DRIFT
# performace degraded and enough labels available       TRIGGER_RETRAINING_EVALUATION
# drift detected, insufficient labels       WAIT_FOR_LABELS

# TRIGGER_RETRAINING_EVALUATION, not DEPLOY_NEW_MODEL
# Retraining creates a candidate. 
# The candidate must still pass evaluation before it can replace the existing model.

# Establish performance policy
# need numerical rules for deciding when performance is unacceptable