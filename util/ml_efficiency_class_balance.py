import sys 
sys.path.append("..")
from src.dataset import Dataset
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.exceptions import NotFittedError
import copy
import warnings

# Suppress LightGBM categorical_feature warning
warnings.filterwarnings("ignore", category=UserWarning, message="categorical_feature keyword has been found*")
warnings.filterwarnings("ignore", category=UserWarning, message="categorical_feature in param dict is overridden*")


RUN_GPU = False

from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, true_positive_rate_difference, true_positive_rate, false_positive_rate_difference

def eq_odd(y_test, y_pred, group_test):
    return true_positive_rate_difference(y_test, y_pred, sensitive_features=group_test)\
                + false_positive_rate_difference(y_test, y_pred, sensitive_features=group_test)
import warnings

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import xgboost as xgb

from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

if RUN_GPU:
    from cuml import RandomForestClassifier, DecisionTreeClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator
from lightgbm import LGBMClassifier


adult_dataset_generator = Dataset("adult")
all_data = adult_dataset_generator.original_dataframe.copy()


from definitions import *

problem_classification = {"metrics":[accuracy_score, f1_score, roc_auc_score],
                      "metric_names":["Accuracy", "F1", "ROC AUC"],
                      "fairness_metrics": [eq_odd],
                      "fairness_metric_names": ["Equalized odds"],
                      "generative_methods": ["tvae", "cart", "smote"],}
                      



# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


categorical_cols = adult_dataset_generator.categorical_input_cols.copy()
categorical_cols.remove("sex")

transformations = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, adult_dataset_generator.continuous_input_cols),
        ('cat', categorical_transformer, categorical_cols)])


# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf_RF = Pipeline(steps=[('preprocessor', transformations),
                      ('classifier', RandomForestClassifier(random_state=42))])
clf_DT = Pipeline(steps=[('preprocessor', transformations),
                    ('classifier', DecisionTreeClassifier(random_state=42))])     


model_names_classification = ["LightGBM", "XGBoost", "Catboost", "Decission Tree", "Random Forest"]


models_classification = [LGBMClassifier, xgb.XGBClassifier, CatBoostClassifier, clf_DT, clf_RF]

args = [{"categorical_feature":adult_dataset_generator.categorical_input_cols, "verbose":-1}, {"enable_categorical":True, "tree_method":'hist'}, {"random_state":42, "loss_function":"Logloss", "verbose":False, "iterations":100, "learning_rate":0.01, "cat_features":adult_dataset_generator.categorical_input_cols}, {}, {}]

problems_classification = []
for model, name, arg in zip(models_classification, model_names_classification, args):
    problem = problem_classification.copy()
    problem["model"] = copy.deepcopy(model)
    problem["model_name"] = name
    problem["args"] = arg
    problems_classification.append(problem)

sampling_method = "class_protected"

average, std, feat_imp_average, feat_imp_std = run_experiments(problems_classification, adult_dataset_generator, all_data, num_repeats = 5, num_folds = 3, protected_attributes = ["sex"], sampling_method=sampling_method)


np.savez('../results/arrays/arrays_{}.npz'.format(sampling_method), average=average, std=std, feat_imp_average=feat_imp_average, feat_imp_std=feat_imp_std)
