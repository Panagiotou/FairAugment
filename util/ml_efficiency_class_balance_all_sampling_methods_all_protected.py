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
import click

class ClickPythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except Exception as e:
            print(e)
            raise click.BadParameter(value)
        
# Suppress LightGBM categorical_feature warning
warnings.filterwarnings("ignore", category=UserWarning, message="categorical_feature keyword has been found*")
warnings.filterwarnings("ignore", category=UserWarning, message="categorical_feature in param dict is overridden*")


RUN_GPU = False




import warnings

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
import xgboost as xgb

from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

if RUN_GPU:
    from cuml import RandomForestClassifier, DecisionTreeClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.base import BaseEstimator
from lightgbm import LGBMClassifier

from definitions import *

@click.command()
@click.option('--dataset', default="adult", help="Dataset", type=str)
def run(dataset):
            
    # if not os.path.exists("../results/{}/".format(dataset):
    #     os.makedirs("../results/{}/".format(dataset)

    if not os.path.exists("../results/{}/arrays/".format(dataset)):
        os.makedirs("../results/{}/arrays/".format(dataset))

    print("Running", dataset)

    dataset_generator = Dataset(dataset)
    dtype_map = dataset_generator.dtype_map
    all_data = dataset_generator.original_dataframe.copy()

    column_types_map = [dataset_generator.dtype_map[col] for col in all_data.columns]

    # Check if all columns have the data type 'category'
    all_categorical = all(dtype == 'category' for dtype in column_types_map)

    # generative_methods = ["gaussian_copula", "ctgan", "tvae", "cart", "smote"]
    generative_methods = ["cart"]
    # generative_methods = []

    if all_categorical:
        print("Only categorical features, dropping SMOTE")
        generative_methods.remove("smote")

    problem_classification = {"metrics":[accuracy_score, f1_score, roc_auc_score],
                        "metric_names":["Acc", "F1", "ROC AUC"],
                        "fairness_metrics": [eq_odd, stat_par, eq_opp],
                        "fairness_metric_names": ["Equalized odds", "Statistical Parity", "Equal Opportunity"],
                        "generative_methods":generative_methods,
                        "sampling_methods":['class', 'class_protected', 'protected', 'same_class']}
                        



    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])


    # categorical_transformer_lgbm = Pipeline(steps=[
    #     ('ordinal', PositiveOrdinalEncoder())
    # ])

    categorical_cols = dataset_generator.categorical_input_cols.copy()

    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, dataset_generator.continuous_input_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # transformations_lgbm = ColumnTransformer(
    #     transformers=[
    #         ('num', numeric_transformer, dataset_generator.continuous_input_cols),
    #         ('cat', categorical_transformer_lgbm, categorical_cols)])
    
    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    clf_RF = Pipeline(steps=[('preprocessor', transformations),
                        ('classifier', RandomForestClassifier(random_state=42))])
    clf_DT = Pipeline(steps=[('preprocessor', transformations),
                        ('classifier', DecisionTreeClassifier(random_state=42))])     

    # clf_lgbm = Pipeline(steps=[('preprocessor', transformations_lgbm),
    #                     ('classifier', LGBMClassifier(categorical_feature=dataset_generator.categorical_input_col_locations, verbose=-1))])  

                        

    # model_names_classification = ["LightGBM"]
    model_names_classification = ["XGBoost"]
    # model_names_classification = ["LightGBM", "XGBoost", "Decission Tree", "Random Forest"]


    models_classification = [xgb.XGBClassifier]
    # models_classification = [LGBMClassifier, xgb.XGBClassifier, clf_DT, clf_RF]



    args = [{"enable_categorical":True, "tree_method":'hist'}]

    problems_classification = []
    for model, name, arg in zip(models_classification, model_names_classification, args):
        problem = problem_classification.copy()
        problem["model"] = copy.deepcopy(model)
        problem["model_name"] = name
        problem["args"] = arg
        problems_classification.append(problem)

    num_repeats = 2
    num_folds = 3


    # protected_attributes_all = ["sex"]
    protected_attributes_all = ["sex", "race", "both"]

    average, std = run_experiments_all_sampling_all_protected(dataset, problems_classification, dataset_generator, all_data,
                    num_repeats = num_repeats, num_folds = num_folds, protected_attributes_in = protected_attributes_all, dtype_map=dtype_map)

    print(average.shape)
    out_file = '../results/{}/arrays/arrays_all_models_all_fairness_metrics_protected_{}_small.npz'.format(dataset, "_".join(protected_attributes_all))
    print(out_file)
    np.savez(out_file, average=average, std=std)

if __name__ == '__main__':
    run()