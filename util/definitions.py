from sklearn.model_selection import cross_val_score, RepeatedKFold

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator
import numpy as np 
import pandas as pd
from catboost import Pool
from sklearn import tree
import graphviz

def generate_latex_table_max(all_metrics_mean, all_metrics_std, names_train, names_test, problems, test_data=False, metric_names_actual=[], metrics_optimal=None, sampling_method=""):
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
        # print(problems[problem_i]["model_name"])
        latex_table += "\\multirow{" + str(1*len(problems)) + "}{*}{" + problems[problem_i]["model_name"] + "}"
        # max_means = [0] * len(names_test)  # Initialize a list to store the maximum mean value for each column
        
        # # Iterate over each test name to find the maximum mean value for each column
        # for j in range(len(names_test)):
        #     max_mean = 0
        #     for i in range(len(names_train)):
        #         avg_metric = all_metrics_mean[problem_i][i][j]
        #         # Check if the current value is greater than the current max_mean
        #         if avg_metric.max() > max_mean:
        #             max_mean = avg_metric.max()
        #     max_means[j] = "{:.3f}".format(max_mean)

        max_of_each_column = np.max(all_metrics_mean[problem_i], axis=0)[0]
        min_of_each_column = np.min(all_metrics_mean[problem_i], axis=0)[0]

        sorted_arr = np.sort(all_metrics_mean[problem_i], axis=0)

        sorted_arr_inv = sorted_arr[::-1]

        # Select the second element in each column
        second_max_of_each_column= sorted_arr_inv[1][0]
        second_min_of_each_column= sorted_arr[1][0]


        max_of_each_column = ["{:.3f}".format(x) for x in max_of_each_column]
        second_max_of_each_column = ["{:.3f}".format(x) for x in second_max_of_each_column]
        second_min_of_each_column = ["{:.3f}".format(x) for x in second_min_of_each_column]

        min_of_each_column = ["{:.3f}".format(x) for x in min_of_each_column]

        for i in range(len(names_train)):
            train_name = names_train[i]
            # if "makecell" in train_name:
            #     latex_table += " & " + "\\multirow{2}{*}{" + train_name + "}"
            # else:
            latex_table += " & " + "\\multirow{1}{*}{" + train_name + "}"
            # avg_metric = all_metrics_mean[metric_row][name_row][metric_col]
            # std_metric = all_metrics_std[metric_row][name_row][metric_col]
            # latex_table += f"& {avg_metric:.3f} ({std_metric:.3f})"
            avgs_stds_c = ""
            # avgs_c = ""
            # stds_c = ""
            for j in range(len(names_test)):
                test_name = names_test[j]
                avg_metric = all_metrics_mean[problem_i][i][j]
                std_metric = all_metrics_std[problem_i][i][j]
                # std_metric = all_metrics_std[metric_row][name_row][metric_col]
                # avgs_c += " & " + " & ".join(map(lambda x: "{:.3f}".format(x), avg_metric))
                # stds_c += " & " + " & ".join(map(lambda x: "({:.3f})".format(x), std_metric))
                avg_metric = ["{:.3f}".format(x) for x in avg_metric]
                std_metric = ["{:.3f}".format(x) for x in std_metric]

                for i in range(len(avg_metric)):
                    if avg_metric[i] == max_of_each_column[i] and metrics_optimal[i]=="max":
                        avg_metric[i] = '\\textbf{' + avg_metric[i] + '}'
                    if avg_metric[i] == min_of_each_column[i] and metrics_optimal[i]=="min":
                        avg_metric[i] = '\\textbf{' + avg_metric[i] + '}'

                    if avg_metric[i] == second_max_of_each_column[i] and metrics_optimal[i]=="max":
                                avg_metric[i] = '\\underline{' + avg_metric[i] + '}'

                    if avg_metric[i] == second_min_of_each_column[i] and metrics_optimal[i]=="min":
                        avg_metric[i] = '\\underline{' + avg_metric[i] + '}'

                # avgs_stds_c += " & " + " & ".join(map(lambda x, y: "{:.3f} \scriptsize{$\pm$ {:.3f}}".format(x, y), avg_metric, std_metric))
                avgs_stds_c += " & " + " & ".join(map(lambda x: "{} \scriptsize{{$\pm$ {}}}".format(x[0], x[1]), zip(avg_metric, std_metric)))

                # if test_data:
                #     latex_table += " & " + test_name + " & " +  numbers + " \\\\\n"
                # else:
            latex_table += avgs_stds_c + " \\\\\n"
            # latex_table += avgs_c + " \\\\\n"
            # latex_table += " & " + stds_c + " \\\\\n"

                # latex_table += "\\cline{2-" + all_cols + "}\n"
        latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    # latex_table += "}\n"
    latex_table += "\\caption{{Comparison, sampling method = {}}}\n".format(sampling_method)
    # latex_table += "\\label{tab:eval}\n"
    latex_table += "\\end{table}"
    
    return latex_table

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
        # print(problems[problem_i]["model_name"])
        latex_table += "\\multirow{" + str(1*len(problems)) + "}{*}{" + problems[problem_i]["model_name"] + "}"

        for i in range(len(names_train)):
            train_name = names_train[i]
            # if "makecell" in train_name:
            #     latex_table += " & " + "\\multirow{2}{*}{" + train_name + "}"
            # else:
            latex_table += " & " + "\\multirow{1}{*}{" + train_name + "}"
            # avg_metric = all_metrics_mean[metric_row][name_row][metric_col]
            # std_metric = all_metrics_std[metric_row][name_row][metric_col]
            # latex_table += f"& {avg_metric:.3f} ({std_metric:.3f})"
            avgs_stds_c = ""
            # avgs_c = ""
            # stds_c = ""
            for j in range(len(names_test)):
                test_name = names_test[j]
                avg_metric = all_metrics_mean[problem_i][i][j]
                std_metric = all_metrics_std[problem_i][i][j]
                # std_metric = all_metrics_std[metric_row][name_row][metric_col]
                # avgs_c += " & " + " & ".join(map(lambda x: "{:.3f}".format(x), avg_metric))
                # stds_c += " & " + " & ".join(map(lambda x: "({:.3f})".format(x), std_metric))

                # avgs_stds_c += " & " + " & ".join(map(lambda x, y: "{:.3f} \scriptsize{$\pm$ {:.3f}}".format(x, y), avg_metric, std_metric))
                avgs_stds_c += " & " + " & ".join(map(lambda x: "{:.3f} \scriptsize{{$\pm$ {:.3f}}}".format(x[0], x[1]), zip(avg_metric, std_metric)))

                # if test_data:
                #     latex_table += " & " + test_name + " & " +  numbers + " \\\\\n"
                # else:
            latex_table += avgs_stds_c + " \\\\\n"
            # latex_table += avgs_c + " \\\\\n"
            # latex_table += " & " + stds_c + " \\\\\n"

                # latex_table += "\\cline{2-" + all_cols + "}\n"
        latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    # latex_table += "}\n"
    latex_table += "\\caption{Comparison}\n"
    latex_table += "\\label{tab:eval}\n"
    latex_table += "\\end{table}"
    
    return latex_table

def compute_metrics(y_test, y_pred, y_probabilities, problem):
    res_metrics = []
    for metric in problem["metrics"]:
        try:
            metric_value = metric(y_test, y_pred)
        except:
            metric_value = metric(y_test, y_probabilities)

        res_metrics.append(metric_value)
    return res_metrics

def compute_fairness_metrics(y_test, y_pred, problem, group):
    return [m(y_test, y_pred, group) for m in problem["fairness_metrics"]]    

def compute_feature_importance(model, model_name, feature_names):
    if model_name=="Catboost":
        fe = model.get_feature_importance(prettified=True).to_dict()

        fei = {fe['Feature Id'][i]: fe['Importances'][i] for i in range(len(fe['Feature Id']))}
        total_score = sum(fei.values())

        feature_importance_dict = {key: value / total_score for key, value in fei.items()}
    elif model_name=="XGBoost":
        fe = model.feature_importances_
        feature_importance_dict = {k: v for k, v in zip(feature_names, fe)}
    elif model_name=="LightGBM":
        fe = model.feature_importances_
        total_importance = np.sum(fe)
        feature_importances_percentage = (fe / total_importance)

        feature_importance_dict = {k: v for k, v in zip(feature_names, feature_importances_percentage)}
    else:
        fn = model[:-1].get_feature_names_out()

        mdi_importances = pd.Series(
            model[-1].feature_importances_, index=fn
        ).sort_values(ascending=True)

        fe = mdi_importances.to_dict()

        feature_importance_dict = {}

        for key, value in fe.items():
            feature_name = key.split('__')[1].split("_")[0]  # Extract the feature name

            if feature_name in feature_importance_dict:
                feature_importance_dict[feature_name] += value  # Add the value if feature already exists
            else:
                feature_importance_dict[feature_name] = value  # Create a new entry if feature doesn't exist


    ordered_list = [feature_importance_dict[key] for key in feature_names]
    
    return ordered_list

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

def train_eval(X_train, y_train, X_test, y_test, problem, keep_protected_input=False, visualize_tree=False, sampling_method="class_protected"):

    if not keep_protected_input:
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        group_test = X_test_copy["sex"].copy()

        X_train = X_train_copy.drop('sex', axis=1)
        X_test = X_test_copy.drop('sex', axis=1)
        if "cat_features" in problem["args"]:
            cflist = [item for item in problem["args"]["cat_features"] if item != "sex"]
            problem["args"]["cat_features"] = cflist
            # problem["args"]["feature_names"] = list(X_train.columns)


    if problem["args"]:
        model = problem["model"](**problem["args"])
    else:
        model = problem["model"]
        
    if is_fitted(model, X_test):
        print("Model is already fitted!")
        return

    X_train[X_train.select_dtypes(['object']).columns] = X_train.select_dtypes(['object']).astype("category")


    fit_inputs = (X_train, y_train)

    if visualize_tree:
        if "cat_features" in problem["args"]: 
            fit_inputs = Pool(X_train, y_train, cat_features=problem["args"]["cat_features"], feature_names=list(X_train.columns))
            problem["args"]["max_depth"] = 4

    if isinstance(fit_inputs, tuple):
        model.fit(*fit_inputs)
    else:
        model.fit(fit_inputs)  # Pass fit_inputs directly


    y_pred = model.predict(X_test)

    if visualize_tree:
        if "cat_features" in problem["args"]: 
            graph = model.plot_tree(tree_idx=0, pool=fit_inputs)
        else:
            # graph = model[-1].export_graphviz()
            dot_data = tree.export_graphviz(model[-1], filled=True, rounded=True, special_characters=True, out_file=None)
            graph = graphviz.Source(dot_data)

        graph.render('../results/tree_viz/graph_{}_{}'.format(problem["model_name"], sampling_method), format='png')


    y_probabilities = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_probabilities, problem)
    fairness_metrics = compute_fairness_metrics(y_test, y_pred, problem, group_test)

    feature_names = X_train.columns.to_list()
    feature_importance = compute_feature_importance(model,  problem["model_name"], feature_names)

    metrics_return = metrics + fairness_metrics
    return metrics_return, y_pred, feature_importance

def get_synthetic_splits(adult_dataset_generator, split_dfs, generative_method="cart", generative_seed=0, return_plot=False, sampling_method="class_protected"):
    max_length_df_key = max(split_dfs, key=lambda x: len(split_dfs[x]))
    # Retrieve the DataFrame using the key
    max_length_df = split_dfs[max_length_df_key]

    max_length_df_class_counts = max_length_df['income'].value_counts()

    max_length_df_majority_class = max_length_df_class_counts.idxmax()
    max_length_df_majority_class_count = max_length_df_class_counts[max_length_df_majority_class]

    total_count = max_length_df_class_counts.sum()  # Summing up the counts
    percentages = (max_length_df_class_counts / total_count) * 100
    max_length_df_class_counts_percentage = percentages
    
    if sampling_method=="class_protected":
        augmented_dfs = []
        if return_plot:
            augmented_dfs_plot = []

        for split_key, split_df in split_dfs.items():
            class_counts = split_df['income'].value_counts()
            augmented_dfs.append(split_df)
            if return_plot:
                split_df_plot = split_df.copy()
                split_df_plot["method"] = "real"
                augmented_dfs_plot.append(split_df_plot.copy())
                
            for class_label, class_count in class_counts.items():
                minority_class_count = class_count
                imbalance = max_length_df_majority_class_count - minority_class_count
                size = imbalance

                if size > 0:
                    class_split_df = split_df[split_df['income'] == class_label].copy()
                    class_split_df.drop('income', axis=1, inplace=True)
                    class_split_df.drop('sex', axis=1, inplace=True)
                    if generative_method=="tvae":
                        split_synthesizer = adult_dataset_generator.train_synthesizer("tvae", class_split_df, encode=False, random_state=generative_seed) 
                        split_synthetic_data = adult_dataset_generator.generate_data(split_synthesizer, num=size, name="tvae", decode=False, random_state=generative_seed)
                    else:
                        split_synthesizer = adult_dataset_generator.train_synthesizer(generative_method, class_split_df, encode=True, random_state=generative_seed) 
                        split_synthetic_data = adult_dataset_generator.generate_data(split_synthesizer, num=size, random_state=generative_seed)
                        
                    split_synthetic_data['income'] = class_label
                    split_synthetic_data['sex'] = split_key
                    augmented_dfs.append(split_synthetic_data.copy())
                    if return_plot:
                        split_synthetic_data_plot = split_synthetic_data.copy()
                        split_synthetic_data_plot['method'] = "synthetic"

                        augmented_dfs_plot.append(split_synthetic_data_plot.copy())
        if return_plot:
            return augmented_dfs, augmented_dfs_plot
        return augmented_dfs

    if sampling_method=="protected":
        max_length_df_key = max(split_dfs, key=lambda x: len(split_dfs[x]))
        max_length_df_length = len(max_length_df)

        augmented_dfs = []
        if return_plot:
            augmented_dfs_plot = []

        for split_key, split_df in split_dfs.items():
            class_counts = split_df['income'].value_counts()
            augmented_dfs.append(split_df)
            if return_plot:
                split_df_plot = split_df.copy()
                split_df_plot["method"] = "real"
                augmented_dfs_plot.append(split_df_plot.copy())
                
            size_current_df = len(split_df)
            imbalance = max_length_df_length - size_current_df
            size = imbalance

            if size > 0:
                class_split_df = split_df.copy()
                class_split_df.drop('sex', axis=1, inplace=True)
                if generative_method=="tvae":
                    split_synthesizer = adult_dataset_generator.train_synthesizer("tvae", class_split_df, encode=False, random_state=generative_seed) 
                    split_synthetic_data = adult_dataset_generator.generate_data(split_synthesizer, num=size, name="tvae", decode=False, random_state=generative_seed)
                else:
                    split_synthesizer = adult_dataset_generator.train_synthesizer(generative_method, class_split_df, encode=True, random_state=generative_seed) 
                    split_synthetic_data = adult_dataset_generator.generate_data(split_synthesizer, num=size, random_state=generative_seed)
                    
                split_synthetic_data['sex'] = split_key
                augmented_dfs.append(split_synthetic_data.copy())
                if return_plot:
                    split_synthetic_data_plot = split_synthetic_data.copy()
                    split_synthetic_data_plot['method'] = "synthetic"

                    augmented_dfs_plot.append(split_synthetic_data_plot.copy())
        if return_plot:
            return augmented_dfs, augmented_dfs_plot
        return augmented_dfs

    if sampling_method=="class":
        augmented_dfs = []
        if return_plot:
            augmented_dfs_plot = []

        for split_key, split_df in split_dfs.items():
            class_counts = split_df['income'].value_counts()
            augmented_dfs.append(split_df)
            if return_plot:
                split_df_plot = split_df.copy()
                split_df_plot["method"] = "real"
                augmented_dfs_plot.append(split_df_plot.copy())

            df_majority_class = class_counts.idxmax()
            df_majority_class_count = class_counts[df_majority_class]

            for class_label, class_count in class_counts.items():
                minority_class_count = class_count
                imbalance = df_majority_class_count - minority_class_count
                size = imbalance

                if size > 0:
                    class_split_df = split_df[split_df['income'] == class_label].copy()
                    class_split_df.drop('income', axis=1, inplace=True)
                    class_split_df.drop('sex', axis=1, inplace=True)
                    if generative_method=="tvae":
                        split_synthesizer = adult_dataset_generator.train_synthesizer("tvae", class_split_df, encode=False, random_state=generative_seed) 
                        split_synthetic_data = adult_dataset_generator.generate_data(split_synthesizer, num=size, name="tvae", decode=False, random_state=generative_seed)
                    else:
                        split_synthesizer = adult_dataset_generator.train_synthesizer(generative_method, class_split_df, encode=True, random_state=generative_seed) 
                        split_synthetic_data = adult_dataset_generator.generate_data(split_synthesizer, num=size, random_state=generative_seed)
                        
                    split_synthetic_data['income'] = class_label
                    split_synthetic_data['sex'] = split_key
                    augmented_dfs.append(split_synthetic_data.copy())
                    if return_plot:
                        split_synthetic_data_plot = split_synthetic_data.copy()
                        split_synthetic_data_plot['method'] = "synthetic"

                        augmented_dfs_plot.append(split_synthetic_data_plot.copy())
        if return_plot:
            return augmented_dfs, augmented_dfs_plot
        return augmented_dfs

    if sampling_method=="same_class":
        augmented_dfs = []
        if return_plot:
            augmented_dfs_plot = []

        for split_key, split_df in split_dfs.items():
            class_counts = split_df['income'].value_counts()
            augmented_dfs.append(split_df)
            if return_plot:
                split_df_plot = split_df.copy()
                split_df_plot["method"] = "real"
                augmented_dfs_plot.append(split_df_plot.copy())
            total_split_count = class_counts.sum()  # Summing up the counts
            split_percentages = (class_counts / total_split_count) * 100
            for class_label, class_percentage in split_percentages.items():
                minority_class_percentage = class_percentage/100
                max_length_df_class_counts_percentage_class = max_length_df_class_counts_percentage[class_label]/100

                class_1_instances = int(class_counts[class_label])
                size = 0
                if not max_length_df_class_counts_percentage_class == minority_class_percentage:
                    additional_class_1_instances = (max_length_df_class_counts_percentage_class * (sum(class_counts)) - class_1_instances) / (1 - max_length_df_class_counts_percentage_class)
                    size = int(additional_class_1_instances)

                if size > 0:
                    class_split_df = split_df[split_df['income'] == class_label].copy()
                    class_split_df.drop('income', axis=1, inplace=True)
                    class_split_df.drop('sex', axis=1, inplace=True)
                    if generative_method=="tvae":
                        split_synthesizer = adult_dataset_generator.train_synthesizer("tvae", class_split_df, encode=False, random_state=generative_seed) 
                        split_synthetic_data = adult_dataset_generator.generate_data(split_synthesizer, num=size, name="tvae", decode=False, random_state=generative_seed)
                    else:
                        split_synthesizer = adult_dataset_generator.train_synthesizer(generative_method, class_split_df, encode=True, random_state=generative_seed) 
                        split_synthetic_data = adult_dataset_generator.generate_data(split_synthesizer, num=size, random_state=generative_seed)
                        
                    split_synthetic_data['income'] = class_label
                    split_synthetic_data['sex'] = split_key
                    augmented_dfs.append(split_synthetic_data.copy())
                    if return_plot:
                        split_synthetic_data_plot = split_synthetic_data.copy()
                        split_synthetic_data_plot['method'] = "synthetic"

                        augmented_dfs_plot.append(split_synthetic_data_plot.copy())
        if return_plot:
            return augmented_dfs, augmented_dfs_plot
        return augmented_dfs

def run_experiments(problems_classification, adult_dataset_generator, all_data, num_repeats = 1, num_folds = 2, protected_attributes = ["sex"], keep_protected_input=False, split_test_set=False, sampling_method="same_class", visualize_tree=False):

    average_problems = []
    std_problems = []
    problems_all = []
    problem_feat_imp_all = []
    rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=42)
    for i, (train_index, test_index) in enumerate(rkf.split(all_data)):    

        print("Split", i, "/", num_repeats*num_folds)
        
        data_train, data_test = all_data.loc[train_index], all_data.loc[test_index]
        data_train_encoded = adult_dataset_generator.encode(data_train, keep_dtypes=True)
        data_test_encoded = adult_dataset_generator.encode(data_test)



        X_train_real = data_train.copy().drop(columns=["income"])

        y_train_real = data_train_encoded["income"].copy().astype("int")

        if split_test_set:
            test_sets, _ = adult_dataset_generator.split_population(data_test)
        else:
            test_sets = {}
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
        # split_df_keys, split_df_vals = zip(*split_dfs.items())


        train_sets_X = [X_train_real]
        train_sets_y = [y_train_real]

        for j, generative_method in enumerate(problems_classification[0]["generative_methods"]):
            generative_seed = hash((i, j)) % (2**32 - 1)

            augmented_dfs =  get_synthetic_splits(adult_dataset_generator, split_dfs, generative_method=generative_method, generative_seed=generative_seed, sampling_method=sampling_method)

            # for split_key, split_df in split_dfs.items():
            #     class_counts = split_df['income'].value_counts()
            #     augmented_dfs.append(split_df)

            #     for class_label, class_count in class_counts.items():
            #         minority_class_count = class_count
            #         imbalance = max_length_df_majority_class_count - minority_class_count
            #         size = imbalance

            #         if size > 0:
            #             class_split_df = split_df[split_df['income'] == class_label].copy()
            #             class_split_df.drop('income', axis=1, inplace=True)
            #             class_split_df.drop('sex', axis=1, inplace=True)
            #             if generative_method=="tvae":
            #                 split_synthesizer = adult_dataset_generator.train_synthesizer("tvae", class_split_df, encode=False, random_state=generative_seed) 
            #                 split_synthetic_data = adult_dataset_generator.generate_data(split_synthesizer, num=size, name="tvae", decode=False, random_state=generative_seed)
            #             else:
            #                 split_synthesizer = adult_dataset_generator.train_synthesizer(generative_method, class_split_df, encode=True, random_state=generative_seed) 
            #                 split_synthetic_data = adult_dataset_generator.generate_data(split_synthesizer, num=size, random_state=generative_seed)
                            
            #             split_synthetic_data['income'] = class_label
            #             split_synthetic_data['sex'] = split_key
            #             augmented_dfs.append(split_synthetic_data.copy())

            augmented_trainingset = pd.concat(augmented_dfs)
            augmented_trainingset_encoded = adult_dataset_generator.encode(augmented_trainingset, keep_dtypes=True)

            X_train_augmented = augmented_trainingset.drop(columns=["income"])
            y_train_augmented = augmented_trainingset_encoded["income"].astype("int")

            # train_real = data_train_encoded["income"].astype("int")
            train_sets_X.append(X_train_augmented)
            train_sets_y.append(y_train_augmented)

        metrics_all = []
        feat_imp_all = []
        for problem in problems_classification:
            metrics_split = []
            feat_imp_split = []
            for X_train, y_train in zip(train_sets_X, train_sets_y):
                setup_metrics = []
                preds = [] 
                fe = [] 
                for test_set_name, test_set in test_sets.items():
                    test_set_encoded = adult_dataset_generator.encode(test_set)
                    X_test = test_set.drop(columns=["income"])
                    y_test = test_set_encoded["income"].astype("int")
                    results, pred, feature_importance = train_eval(X_train, y_train, X_test, y_test, problem, keep_protected_input=keep_protected_input, visualize_tree=visualize_tree, sampling_method=sampling_method)
                    setup_metrics.append(results)
                    preds.append(pred)
                    fe.append(feature_importance)
                metrics_split.append(setup_metrics)
                feat_imp_split.append(fe)
            metrics_all.append(metrics_split)
            feat_imp_all.append(feat_imp_split)
        problems_all.append(metrics_all)
        problem_feat_imp_all.append(feat_imp_all)

    problems_all = np.array(problems_all)

    problem_feat_imp_all = np.array(problem_feat_imp_all)


    average_metrics_all = np.mean(problems_all, axis=0)
    std_metrics_all = np.std(problems_all, axis=0)

    average_feature_importance_all = np.mean(problem_feat_imp_all, axis=0)
    std_feature_importance_all = np.std(problem_feat_imp_all, axis=0)

    return average_metrics_all, std_metrics_all, average_feature_importance_all, std_feature_importance_all


    
# def run_experiments_old(problems_classification, adult_dataset_generator, all_data, num_repeats = 1, num_folds = 2, protected_attributes = ["sex"], keep_protected_input=False):

#     average_problems = []
#     std_problems = []
#     for problem in problems_classification:
#         print(problem["model_name"])

#         rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=42)
#         all_metrics_mean = []
#         all_metrics_std = []
#         metrics_all = []
#         for i, (train_index, test_index) in enumerate(rkf.split(all_data)):    

#             print(problem["model_name"], i)
#             data_train, data_test = all_data.loc[train_index], all_data.loc[test_index]
#             data_train_encoded = adult_dataset_generator.encode(data_train, keep_dtypes=True)
#             data_test_encoded = adult_dataset_generator.encode(data_test)


#             X_train_real = data_train.copy().drop(columns=["income"])
#             y_train_real = data_train_encoded["income"].copy().astype("int")

#             test_sets, _ = adult_dataset_generator.split_population(data_test)
#             test_sets["all"] = data_test

#             split_dfs, additional_sizes = adult_dataset_generator.split_population(data_train, protected_attributes=protected_attributes)


#             # Get the DataFrame with the maximum length
#             max_length_df_key = max(split_dfs, key=lambda x: len(split_dfs[x]))
#             # Retrieve the DataFrame using the key
#             max_length_df = split_dfs[max_length_df_key]

#             max_length_df_class_counts = max_length_df['income'].value_counts()

#             max_length_df_majority_class = max_length_df_class_counts.idxmax()
#             max_length_df_majority_class_count = max_length_df_class_counts[max_length_df_majority_class]

#             augmented_dfs = []
#             split_df_keys, split_df_vals = zip(*split_dfs.items())


#             train_sets_X = [X_train_real]
#             train_sets_y = [y_train_real]

#             for j, generative_method in enumerate(problem["generative_methods"]):
#                 generative_seed = hash((i, j)) % (2**32 - 1)
#                 print("\t", generative_method, "seed", generative_seed)

#                 for split_key, split_df in split_dfs.items():
#                     class_counts = split_df['income'].value_counts()
#                     augmented_dfs.append(split_df)

#                     for class_label, class_count in class_counts.items():
#                         minority_class_count = class_count
#                         imbalance = max_length_df_majority_class_count - minority_class_count
#                         size = imbalance

#                         if size > 0:
#                             class_split_df = split_df[split_df['income'] == class_label].copy()
#                             class_split_df.drop('income', axis=1, inplace=True)
#                             class_split_df.drop('sex', axis=1, inplace=True)
#                             if generative_method=="tvae":
#                                 split_synthesizer = adult_dataset_generator.train_synthesizer("tvae", class_split_df, encode=False, random_state=generative_seed) 
#                                 split_synthetic_data = adult_dataset_generator.generate_data(split_synthesizer, num=size, name="tvae", decode=False, random_state=generative_seed)
#                             else:
#                                 split_synthesizer = adult_dataset_generator.train_synthesizer(generative_method, class_split_df, encode=True, random_state=generative_seed) 
#                                 split_synthetic_data = adult_dataset_generator.generate_data(split_synthesizer, num=size, random_state=generative_seed)
                                
#                             split_synthetic_data['income'] = class_label
#                             split_synthetic_data['sex'] = split_key
#                             augmented_dfs.append(split_synthetic_data.copy())

#                 augmented_trainingset = pd.concat(augmented_dfs)
#                 augmented_trainingset_encoded = adult_dataset_generator.encode(augmented_trainingset, keep_dtypes=True)

#                 X_train_augmented = augmented_trainingset.drop(columns=["income"])
#                 y_train_augmented = augmented_trainingset_encoded["income"].astype("int")

#                 # train_real = data_train_encoded["income"].astype("int")
#                 train_sets_X.append(X_train_augmented)
#                 train_sets_y.append(y_train_augmented)

#             metrics_split = []
            
#             for X_train, y_train in zip(train_sets_X, train_sets_y):
#                 setup_metrics = []
#                 preds = [] 
#                 for test_set_name, test_set in test_sets.items():
#                     test_set_encoded = adult_dataset_generator.encode(test_set)
#                     X_test = test_set.drop(columns=["income"])
#                     y_test = test_set_encoded["income"].astype("int")
#                     results, pred = train_eval(X_train, y_train, X_test, y_test, problem, keep_protected_input=keep_protected_input)
#                     setup_metrics.append(results)
#                     preds.append(pred)
#                 metrics_split.append(setup_metrics)
#             print("metrics_split", np.array(metrics_split).shape)
#             metrics_all.append(metrics_split)
#         metrics_all = np.array(metrics_all)    
#         print("Metrics_all", metrics_all.shape)
#         average_metrics_all = np.mean(metrics_all, axis=0)
#         std_metrics_all = np.std(metrics_all, axis=0)
#         average_problems.append(average_metrics_all)
#         std_problems.append(std_metrics_all)
#     return np.array(average_problems), np.array(std_problems)
