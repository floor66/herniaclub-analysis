# herniaclub-analysis
Parsing and analysis for the Hernia Club database

ONLY HEADERS.xlsx contains the variables and descriptions of variables of the entire dataset (available upon reasonable request, omitted for privacy reasons)

conversion.py converts the raw .xlsx database dump into a SPSS-readable (.sav) format with proper variable encoding of the dozens of variables of different types present in the database.

analysis.py uses fast.ai in order to perform analyses on the dataset (XGBoost and "simple" logistic regression)
