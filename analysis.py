# Hernia Club database analysis program
# Floris den Hartog and Rudolf van den Berg, Feb 2023

# %% Imports and definitions
import random
import torch
from math import sqrt

import pandas as pd
import pyreadstat
import numpy as np
from fastai.tabular.all import *
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel
import missingno as msno
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay


# Set up constants
NUM_WORKERS = 0
RANDOM_SEED = 42
DATABASE_PATH = (r'Z:\HC'
                 r'\Db_2022_conversie.sav')

# A SMOTE_RATIO > 0 will apply oversampling to the training set
# Not sure if oversampling makes sense yet because in practice, groups are
# likely to be imbalanced too
SMOTE_RATIO = 0.5

TEST_SPLIT = 0.2

# Define some helper functions
def roc_auc_95ci(y_true, y_score, positive=1):
    score = roc_auc_score(y_true, y_score)
    n_1 = sum(y_true == positive)
    n_2 = sum(y_true != positive)
    q_1 = score / (2 - score)
    q_2 = 2 * score**2 / (1 + score)
    standard_error = sqrt((score*(1 - score) + (n_1 - 1)*(q_1 - score**2) + (n_2 - 1)*(q_2 - score**2)) / (n_1 * n_2))
    lower_bound = score - 1.96 * standard_error
    upper_bound = score + 1.96 * standard_error

    if lower_bound < 0:
        lower_bound = 0
    if upper_bound > 1:
        upper_bound = 1

    return lower_bound, score, upper_bound

# Loads the SPSS file, creates ordinal categories where needed
def process_db(path):
    data, meta = pyreadstat.read_sav(path)

    def ordered_cat(col, definitions):
        data[data[col] == ""] = np.nan
        data[col] = data[col].astype("category")
        data[col] = data[col].cat.set_categories(definitions, ordered=True)

    ordered_cat("ASA", ("ASA1", "ASA2", "ASA3", "ASA4", "ASA5",))
    ordered_cat("Y1_PAIN_CATEGORY", (0.0, 1.0, 2.0, 3.0,))

    return data

# Set a constant randomness seed so we can reproduce results
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Setup pyTorch so we get reproducable results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Setup pandas, prevents it throwing a SettingWithCopyError when using inplace=True
pd.options.mode.chained_assignment = None

# Open & import the entire database
herniaclub_raw = process_db(DATABASE_PATH)

# %% Select features and target(s), oversample if needed

# Open & import the entire database
herniaclub = herniaclub_raw.copy()

# Select only incisional hernia cases (includes recurrences!)
# herniaclub = herniaclub.loc[herniaclub["Repairedhernia"] == "Incisional hernia (IH) - IVHR (incisional ventral hernia repair)"]

# Specify which input and output variables to include in the model
categorical_input_vars = [
    "ASA", "Gender",
    "MULTI_SITE_REPAIR", "RECURRENT_IH_SURGERY", "Incarceratedhernia", "Emergencysetting",
    "NAROPIN_*",  # Local anaesthetic applied in any form?
    "Surgicaltechnicfinallyperformed",
    "PREOP_HERNIA_SYMPTOM_*",  # Any preoperative hernia-related symptoms?
    "Smoking", "Workingcondition", "Regularphysicalsportactivities",
    "HISTORY_*", "HEALING_FACTOR_*", "DISSECTION_FACTOR_*",
]

continuous_input_vars = [
    "Age", # "HEIGHT_BASELINE", "WEIGHT_BASELINE",
]

output_vars = [
    "POSTOP_30D_SSI",
]

# Parse wildcards in categorical variable names
to_remove = []
for cat_var in categorical_input_vars:
    if "_*" in cat_var:
        to_remove.append(cat_var)
        query = cat_var.replace("_*", "_")
        target_cols = [col for col in herniaclub.columns if query in col and "NONE" not in col]
        categorical_input_vars = categorical_input_vars + target_cols

for cat_var in to_remove:
    categorical_input_vars.pop(categorical_input_vars.index(cat_var))

categorical_input_vars_orig = categorical_input_vars.copy()

# Exclude cases which are missing one or more outcomes
for output_var in output_vars:
    herniaclub = herniaclub[~herniaclub[output_var].isna()]
    if type(herniaclub[output_var].dtype) is not pd.CategoricalDtype:
        herniaclub[output_var] = herniaclub[output_var].astype("category")

# For ASA multiclass testing only: Remove ASA5 because it is too rare
# herniaclub = herniaclub.loc[herniaclub[output_vars[0]] != "ASA5"]
# herniaclub[output_vars[0]] = herniaclub[output_vars[0]].cat.remove_unused_categories()

# Set True if output_var is multi-class
# Single-label targets are True / False (NaN/missing has been removed)
is_multi_class = len(herniaclub[output_vars[0]].unique()) > 2

# Set True if we are performing multi-label classification
is_multi_label = len(output_vars) > 1

# Less human-readable but more conventional variable names
Xs = categorical_input_vars + continuous_input_vars
ys = output_vars

# Select only the columns we need from the dataset
herniaclub = herniaclub[Xs + ys]

# Display missingness in data visually
msno.matrix(herniaclub)

# Stratified split of data into training and validation sets
splits = TrainTestSplitter(test_size=TEST_SPLIT, random_state=RANDOM_SEED,
    stratify=herniaclub[ys])(range_of(herniaclub))

# Randomly split into training/validation data
# splits = RandomSplitter(valid_pct=TEST_SPLIT, seed=RANDOM_SEED)(range_of(herniaclub))

# %% Address imbalance
# Display target balance in the training portion of the original dataset
print("Target balance:")
print("Original set\n", herniaclub.iloc[splits[0]].value_counts(ys), "\n")

if SMOTE_RATIO > 0:
    # Preprocess the entire dataset first
    tab_smote = TabularPandas(
        df=herniaclub,
        procs=[Categorify, FillMissing, Normalize],
        cat_names=categorical_input_vars,
        cont_names=continuous_input_vars,
        splits=splits,
        inplace=True)

    # SMOTE _ONLY_ the training dataset, leave the validation set as it is
    train_set = tab_smote.train.items.copy()
    valid_set = tab_smote.valid.items.copy()

    # Type of SMOTE that supports categorical variables
    # We need to pass column indices of the categorical columns we would like to include
    smote = SMOTENC(
        categorical_features=[train_set[categorical_input_vars].columns.get_loc(col) for col in categorical_input_vars],
        random_state=RANDOM_SEED,
        sampling_strategy=SMOTE_RATIO)

    # Perform oversampling
    X_sm, y_sm = smote.fit_resample(train_set[Xs], train_set[ys])

    # Create a new df containing the oversampled data
    train_set_oversampled = pd.DataFrame(X_sm)
    train_set_oversampled[ys] = y_sm

    # Concat the _oversampled_ training set + the _original_ validation set
    full_set = pd.concat([train_set_oversampled, valid_set])
    full_set = full_set.reset_index()
    full_set = full_set[Xs + ys]

    # Create splits according to already created train and valid sets
    # Rows 1..n (where n is the amount of cases in the oversampled set) are the training split
    # Rows n+1..n+m (where m is the amount of cases in the validation set) are the validation split
    splits = list(range(len(train_set_oversampled))), \
        [len(train_set_oversampled) + i for i in range(len(valid_set))]

    # We only need to Categorify as this data has already been imputed and normed
    procs = [Categorify]
    
    # Display the target balance of the oversampled dataset
    print("Oversampled set\n", train_set_oversampled.value_counts(ys))
else:
    # No smote, just the full (unprocessed!) dataset with original splits
    full_set = herniaclub.copy()

    # We still need to preprocess this data
    procs = [Categorify, FillMissing, Normalize]

# %%
# Set the right y_block depending on whether we are doing single- or multi-label classification
# Note, if multi-label, make sure targets are already one-hot encoded
y_block = MultiCategoryBlock(encoded=True, vocab=ys) if is_multi_label else CategoryBlock

# %% Log reg
if False:
    logreg_set = herniaclub.copy()
    train_set, valid_set = logreg_set.iloc[splits[0]], logreg_set.iloc[splits[1]]

    one_hot = OneHotEncoder()
    scaler = StandardScaler()
    transformer = ColumnTransformer([("one_hot", one_hot, categorical_input_vars_orig),
                                    ("standard_scaler", scaler, continuous_input_vars)],
                                    remainder="passthrough")

    X_train, y_train = transformer.fit_transform(train_set[Xs]), train_set[ys[0]]
    X_test, y_test = transformer.transform(valid_set[Xs]), valid_set[ys[0]]

    clf = LogisticRegression(random_state=RANDOM_SEED,
                            solver="newton-cholesky")
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    roc_auc_95ci(y_test, proba[:, 1])
    cm = ConfusionMatrixDisplay(confusion_matrix(y_test, clf.predict(X_test)))
    cm.plot()
    fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.title("ROC")
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

# %% XGBoost
if False:
    xgb_set = herniaclub.copy()

    xgb_set[ys[0]] = xgb_set[ys[0]].astype("bool")

    for var in categorical_input_vars_orig:
        xgb_set[var] = xgb_set[var].astype("category") if type(xgb_set[var]) is not pd.CategoricalDtype else xgb_set[var]

    train_set, valid_set = xgb_set.iloc[splits[0]], xgb_set.iloc[splits[1]]

    clf = xgb.XGBClassifier(tree_method="gpu_hist", enable_categorical=True,
                            early_stopping_rounds=25,
                            eval_metric=["auc", "error", "logloss"])

    clf.fit(train_set[Xs], train_set[ys],
            eval_set=[(valid_set[Xs], valid_set[ys])])

    Y_pred = clf.predict(valid_set[Xs])

    cm = ConfusionMatrixDisplay(confusion_matrix(valid_set[ys[0]], Y_pred))
    cm.plot()

    Y_pred = clf.predict_proba(valid_set[Xs])
    y_scores = Y_pred[:, 1]
    fpr, tpr, _ = roc_curve(valid_set[ys], y_scores)
    roc_auc = auc(fpr, tpr)

    plt.title("ROC")
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    print(roc_auc)
    roc_auc_95ci(valid_set[ys[0]], y_scores)

# %%
# New TabularPandas with oversampled data if we are oversampling, else just normal data
tab = TabularPandas(
    df=full_set,
    procs=procs,
    cat_names=categorical_input_vars_orig,
    cont_names=continuous_input_vars,
    y_names=ys,
    y_block=y_block,
    splits=splits,
    inplace=True)

# %% Set network up
if torch.cuda.is_available():
    dl = tab.dataloaders(bs=16, num_workers=NUM_WORKERS, device=torch.device("cuda"))
else:
    dl = tab.dataloaders(bs=16, num_workers=NUM_WORKERS)
dl.show_batch()

config = tabular_config(ps=[0.001, 0.01], embed_p=0.4)
if is_multi_class:
    metrics = [RocAuc(multi_class="ovr"), accuracy]
else:
    if is_multi_label:
        metrics = [RocAucMulti(), RocAucMulti(average=None), PrecisionMulti(), accuracy_multi]
    else:
        metrics = [RocAucBinary(), Precision(), accuracy, F1Score(), BrierScore()]

network = tabular_learner(
    dls=dl,
    layers=[500, 250],
    config=config,
    metrics=metrics)

# network.summary()

# %% Find a learning rate
lrs = network.lr_find(suggest_funcs=(minimum, steep, valley, slide))
print(lrs)

# %% Train the model
network.fit_one_cycle(5, lr_max=lrs.valley)
network.save("HC_model_1")
network.export("HC_model_1")

# %% Assess performance
network.load("HC_model_1")
valid_probas, valid_targets, valid_preds = network.get_preds(dl=dl.valid, with_decoded=True)

if is_multi_class:
    print("ROC_AUC", roc_auc_score(valid_targets, valid_probas, multi_class="ovr"))
else:
    if is_multi_label:
        [print("ROC_AUC", output_vars[i], roc_auc_score(valid_targets[:, i], valid_probas[:, i])) for i in range(len(output_vars))]
    else:
        print("ROC_AUC", output_vars[0], roc_auc_score(valid_targets, valid_probas[:, 1]))

# %% Confusion matrix
interp = ClassificationInterpretation.from_learner(network)
interp.plot_confusion_matrix(normalize=False, title="Confusion matrix"+ " - %s" % output_vars[0] if len(output_vars) == 1 else "")

cm = ConfusionMatrixDisplay(confusion_matrix(dl.valid_ds[ys[0]], valid_preds))
cm.plot()

# %% Display ROC curve
fpr, tpr, _ = roc_curve(valid_targets, valid_probas[:, 1])
plt.title("ROC")
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

roc_auc_95ci(valid_targets, valid_probas[:, 1])

# %%
