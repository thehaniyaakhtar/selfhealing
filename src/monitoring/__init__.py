# enforce the expected 30 input features
# Allow class as an optional label, nver treat as model input
# Reject missing cols
# Reject null, non numeric infinite values
# Validate amount and time are non negative
# Validate labels, if present are 0 or 1
# Return feature in a consistent column order