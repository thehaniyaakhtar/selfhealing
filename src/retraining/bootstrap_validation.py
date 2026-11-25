# paired bootstrap validation
# what is bootstrap validation: statistical resampling technique
# instead of trusting one evaluation repeatedly create new samples from those transactions, allow observations to be selected more than once
# For each sample:
# delta PR AUC = PR AUC (candidate) - PR AUC (incumbent)

# Paired bootstrap:
# both models were tested on the same transaction
# preserve the pairing when resampling
# no independent sampling of two models predictions

# process followed:
# generate a random sampe of transaction indices
# select same indices for both models and the acctual labels
# Calculate both PR AUC scores
# subtract incumbent PR AUC from candidate ones
# Repeat 1000 times
# calculate 95% CI of the differences 

# 95% CI of the mean is a range with upper and lower number calculated from a sample
# True population mean is unknown, range describes the possible values the meean could be

# Entire interval above 0       Evidence that candidate PR AUC is higher
# Interval crosses 0        Improvement in inconclusive
# Entire interval is below 0        Evidence that candidate PR AUC is lower