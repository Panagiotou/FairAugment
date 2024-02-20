import sys 
sys.path.append("..")
from src.dataset import Dataset
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.exceptions import NotFittedError
import copy

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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator

adult_dataset_generator = Dataset("adult")
all_data = adult_dataset_generator.original_dataframe.copy()

def compute_metrics(y_test, y_pred, problem):
    return [m(y_test, y_pred) for m in problem["metrics"]]

def compute_fairness_metrics(y_test, y_pred, problem, group):
    return [m(y_test, y_pred, group) for m in problem["fairness_metrics"]]    

def is_fitted(estimator, X_test):
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            try:
                step.__dict__['estimators_']
            except (AttributeError, KeyError):
                return False
    elif isinstance(estimator, BaseEstimator):
        try:
            estimator.__dict__['estimators_']
        except (AttributeError, KeyError):
            return False
    else:
        try:
            estimator.predict(X_test)
            return True
        except:
            return False
    
    return True

def train_eval(X_train, y_train, X_test, y_test, problem, keep_protected_input=False):

    if not keep_protected_input:
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        group_test = X_test_copy["sex"].copy()

        X_train = X_train_copy.drop('sex', axis=1)
        X_test = X_test_copy.drop('sex', axis=1)
        if "cat_features" in problem["args"]:
            cflist = [item for item in problem["args"]["cat_features"] if item != "sex"]
            problem["args"]["cat_features"] = cflist


    if problem["args"]:
        model = problem["model"](**problem["args"])
    else:
        model = problem["model"]
        
    if is_fitted(model, X_test):
        print("Model is already fitted!")
        return

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, problem)
    fairness_metrics = compute_fairness_metrics(y_test, y_pred, problem, group_test)
    metrics_return = metrics + fairness_metrics
    return metrics_return, y_pred

def run_experiments(problems_classification, adult_dataset_generator, all_data, num_repeats = 1, num_folds = 2, protected_attributes = ["sex"], keep_protected_input=False):

    average_problems = []
    std_problems = []
    for problem in problems_classification:
        print(problem["model_name"])

        rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=42)
        all_metrics_mean = []
        all_metrics_std = []
        metrics_all = []
        for i, (train_index, test_index) in enumerate(rkf.split(all_data)):    

            data_train, data_test = all_data.loc[train_index], all_data.loc[test_index]
            data_train_encoded = adult_dataset_generator.encode(data_train, keep_dtypes=True)
            data_test_encoded = adult_dataset_generator.encode(data_test)


            X_train_real = data_train.copy().drop(columns=["income"])
            y_train_real = data_train_encoded["income"].copy().astype("int")

            test_sets, _ = adult_dataset_generator.split_population(data_test)
            test_sets["all"] = data_test

            split_dfs, additional_sizes = adult_dataset_generator.split_population(data_train, protected_attributes=protected_attributes)


            # Get the DataFrame with the maximum length
            max_length_df_key = max(split_dfs, key=lambda x: len(split_dfs[x]))
            # Retrieve the DataFrame using the key
            max_length_df = split_dfs[max_length_df_key]

            max_length_df_class_counts = max_length_df['income'].value_counts()

            max_length_df_majority_class = max_length_df_class_counts.idxmax()
            max_length_df_majority_class_count = max_length_df_class_counts[max_length_df_majority_class]

            augmented_dfs = []
            split_df_keys, split_df_vals = zip(*split_dfs.items())


            train_sets_X = [X_train_real]
            train_sets_y = [y_train_real]

            for generative_method in problem["generative_methods"]:
                for split_key, split_df in split_dfs.items():
                    class_counts = split_df['income'].value_counts()
                    augmented_dfs.append(split_df)

                    for class_label, class_count in class_counts.items():
                        minority_class_count = class_count
                        imbalance = max_length_df_majority_class_count - minority_class_count
                        size = imbalance

                        if size > 0:
                            class_split_df = split_df[split_df['income'] == class_label].copy()
                            class_split_df.drop('income', axis=1, inplace=True)
                            class_split_df.drop('sex', axis=1, inplace=True)
                            split_synthesizer = adult_dataset_generator.train_synthesizer(generative_method, class_split_df, encode=True) 
                            split_synthetic_data = adult_dataset_generator.generate_data(split_synthesizer, num=size)
                            split_synthetic_data['income'] = class_label
                            split_synthetic_data['sex'] = split_key
                            augmented_dfs.append(split_synthetic_data.copy())

                augmented_trainingset = pd.concat(augmented_dfs)
                augmented_trainingset_encoded = adult_dataset_generator.encode(augmented_trainingset, keep_dtypes=True)

                X_train_augmented = augmented_trainingset.drop(columns=["income"])
                y_train_augmented = augmented_trainingset_encoded["income"].astype("int")

                # train_real = data_train_encoded["income"].astype("int")
                train_sets_X.append(X_train_augmented)
                train_sets_y.append(y_train_augmented)

            metrics_split = []
            
            for X_train, y_train in zip(train_sets_X, train_sets_y):
                setup_metrics = []
                preds = [] 
                for test_set_name, test_set in test_sets.items():
                    test_set_encoded = adult_dataset_generator.encode(test_set)
                    X_test = test_set.drop(columns=["income"])
                    y_test = test_set_encoded["income"].astype("int")
                    results, pred = train_eval(X_train, y_train, X_test, y_test, problem, keep_protected_input=keep_protected_input)
                    setup_metrics.append(results)
                    preds.append(pred)
                metrics_split.append(setup_metrics)
            metrics_all.append(metrics_split)
        metrics_all = np.array(metrics_all)    
        average_metrics_all = np.mean(metrics_all, axis=0)
        std_metrics_all = np.std(metrics_all, axis=0)
        average_problems.append(average_metrics_all)
        std_problems.append(std_metrics_all)
    return np.array(average_problems), np.array(std_problems)



problem_classification = {"metrics":[accuracy_score,  precision_score, recall_score, f1_score],
                      "metric_names":["Accuracy", "P", "R", "F1"],
                      "fairness_metrics": [eq_odd],
                      "fairness_metric_names": ["Equalized odds"],
                      "generative_methods": ["cart", "smote"],}
                      



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


# models = [MultiOutputRegressor(LGBMRegressor(random_state=42)), DecisionTreeRegressor(random_state=42), RandomForestRegressor(random_state=42)]
# models_classification = [xgb.XGBClassifier, CatBoostClassifier, DecisionTreeClassifier, RandomForestClassifier]
# models_classification = [xgb.XGBClassifier]
models_classification = [CatBoostClassifier, clf_DT, clf_RF]

# args = [{"random_state":42}, {"random_state":42, "loss_function":"MultiRMSE", "verbose":False, "iterations":100, "learning_rate":0.01}, {"random_state":42}, {"random_state":42}]
args = [{"random_state":42, "loss_function":"Logloss", "verbose":False, "iterations":100, "learning_rate":0.01, "cat_features":adult_dataset_generator.categorical_input_cols}, {}, {}]

# model_names_classification = ["xgboost", "catboost", "DT", "RF"]
model_names_classification = ["Catboost", "Decission Tree", "Random Forest"]
problems_classification = []
for model, name, arg in zip(models_classification, model_names_classification, args):
    problem = problem_classification.copy()
    problem["model"] = copy.deepcopy(model)
    problem["model_name"] = name
    problem["args"] = arg
    problems_classification.append(problem)

average, std = run_experiments(problems_classification, adult_dataset_generator, all_data, num_repeats = 5, num_folds = 3, protected_attributes = ["sex"])

def generate_latex_table1(all_metrics_mean, all_metrics_std, names_train, names_test, problems, test_data=False, metric_names_actual=[]):
    if test_data:
        all_cols =  str(2 + len(metric_names_actual) * len(names_test))
    else:
        all_cols = str(len(problems[0]["metric_names"]) + 2)
    latex_table = "\\begin{table}[h]\n"
    latex_table += "\\centering\n"
    # latex_table += "\\scalebox{0.70}{\n"
    latex_table += "\\begin{tabular}{l l " + " ".join(["c"]*(int(all_cols)-2)) + "}\n"
    latex_table += "\\hline\n"
    if test_data:
        if len(metric_names_actual) > 0:
            latex_table += "Model & Train data & \multicolumn{" + str(len(names_test) * len(metric_names_actual)) + "}{c}{Test data} \\\\\n" 
            latex_table += "& "
            for name_t in names_test:
                latex_table +=" & \multicolumn{" + str(len(metric_names_actual)) + "}{c}{" + name_t + "}"
            latex_table += "\\\\\n"
            latex_table += "\cline{3-" + str(all_cols) +"}"
            # latex_table +=  "& & " + " & ".join(metric_names_actual) + " & " + " & ".join(metric_names_actual) + " \\\\\n"
            latex_table +=  "& " + "".join([" & " + " & ".join(metric_names_actual) for _ in range(len(names_test))]) + " \\\\\n"
        else:
            latex_table += "Model & Train data & Test data & " + " & ".join(problems[0]["metric_names"]) + " \\\\\n"
    else:
        if len(metric_names_actual) > 0:
            latex_table += "Model & Train data & " + " & ".join(metric_names_actual) + " \\\\\n"
        else:
            latex_table += "Model & Train data & " + " & ".join(problems[0]["metric_names"]) + " \\\\\n"

    latex_table += "\\hline"
    count_make_cell = sum("makecell" in item for item in names_train)

    for problem_i in range(len(problems)):
        print(problems[problem_i]["model_name"])
        latex_table += "\\multirow{" + str(2*len(problems)) + "}{*}{" + problems[problem_i]["model_name"] + "}"

        for i in range(len(names_train)):
            train_name = names_train[i]
            # if "makecell" in train_name:
            #     latex_table += " & " + "\\multirow{2}{*}{" + train_name + "}"
            # else:
            latex_table += " & " + "\\multirow{2}{*}{" + train_name + "}"
            # avg_metric = all_metrics_mean[metric_row][name_row][metric_col]
            # std_metric = all_metrics_std[metric_row][name_row][metric_col]
            # latex_table += f"& {avg_metric:.3f} ({std_metric:.3f})"
            avgs_c = ""
            stds_c = ""
            for j in range(len(names_test)):
                test_name = names_test[j]
                avg_metric = all_metrics_mean[problem_i][i][j]
                std_metric = all_metrics_std[problem_i][i][j]
                # std_metric = all_metrics_std[metric_row][name_row][metric_col]
                avgs_c += " & " + " & ".join(map(lambda x: "{:.3f}".format(x), avg_metric))
                stds_c += " & " + " & ".join(map(lambda x: "({:.3f})".format(x), std_metric))

                # if test_data:
                #     latex_table += " & " + test_name + " & " +  numbers + " \\\\\n"
                # else:
            latex_table += avgs_c + " \\\\\n"
            latex_table += " & " + stds_c + " \\\\\n"

                # latex_table += "\\cline{2-" + all_cols + "}\n"
        latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    # latex_table += "}\n"
    latex_table += "\\caption{Comparison}\n"
    latex_table += "\\label{tab:eval}\n"
    latex_table += "\\end{table}"
    
    return latex_table

metric_names_actual = ["Accuracy", "P", "R", "F1", "Equalized Odds"]
names_train = ["Adult", "Augmented Adult (CART)", "Augmented Adult (SMOTENC)"]
test_sets, _ = adult_dataset_generator.split_population(all_data)
protected_attributes = ["Sex"]
# names_test = ["\\makecell[c]{" + '\\\\'.join([f"{attr}-{value}" for attr, value in zip(protected_attributes, entry)]) + "}" for entry in test_sets.keys()]
# names_test = [' \& '.join([f"{attr}={value}" for attr, value in zip(protected_attributes, entry)]) for entry in test_sets.keys()]
names_test = [f"Sex={value}" for value in test_sets.keys()]
names_test.append("Overall")
latex_table = generate_latex_table1(average, std, names_train, names_test, problems_classification, metric_names_actual=metric_names_actual, test_data=True)
print(latex_table)
