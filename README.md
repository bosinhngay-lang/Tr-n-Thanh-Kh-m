Upload your CSV (rows = students, columns = features + target).
The app auto-detects the target; adjust if needed. Continuous targets are auto-binarized with a balanced threshold.
Adjust sidebar parameters (with tooltips).
Review Descriptive → class balance & correlations.
Train Logistic (MLE) and Bayesian (Laplace or PyMC).
Use Threshold & Gains to tune the decision threshold, and Predict to try custom inputs.
Tip: For imbalanced data, start with class_weight='balanced' and tune the threshold for your business goal (e.g., recall).
