import os
import pandas as pd
from synthpop import Synthpop
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
from .our_smote import SMOTENC_GENERATIVE
import numpy as np
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

from scipy.spatial.distance import pdist, squareform
from sklearn.pipeline import Pipeline

from definitions import PositiveOrdinalEncoder

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer


class Dataset():
    def __init__(self, dataset_name, binary_features=False, ignore_features=[], protected_attribute="sex"):
        self.binary_features = binary_features
        self.ignore_features = ignore_features
        self.protected_attribute = protected_attribute


        dataframe = self.load_dataset(dataset_name)


    def encode(self, df, keep_dtypes=False):
        df_encoded = df.copy()
        for column in df_encoded.columns:
            dtype = self.dtype_map[column]
            if dtype == "category":
                df_encoded[column] = df_encoded[column].map(self.reverse_mappings[column]).astype("category")
        # if keep_dtypes:
        #     df_encoded = df_encoded.astype(df.dtypes)
        return df_encoded

    def decode(self, df, keep_dtypes=False):
        df_decoded = df.copy()
        for column, mapping in self.original_mappings.items():
            if column in df_decoded.columns:
                df_decoded[column] = df_decoded[column].map(mapping)
        if keep_dtypes:
            df_decoded = df_decoded.astype(df.dtypes)
        return df_decoded

    def load_dataset(self, dataset_name):

        if dataset_name == "adult":
            self.ignore_features += ["fnlwgt", "educational-num"]
            url = "https://raw.githubusercontent.com/tailequy/fairness_dataset/main/experiments/data/adult-clean.csv"

            target = "Class-label"

            folder_path = "datasets/{}".format(dataset_name)

            if len(self.ignore_features) > 0:
                dataset_name += "_" + "_".join(self.ignore_features)

            json_file_path = os.path.join(folder_path, "{}.json".format(dataset_name))

            # Check if the folder exists, if not, create it
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Check if the JSON file exists locally
            if not os.path.isfile(json_file_path):
                # File does not exist locally, download the CSV dataset
                df = pd.read_csv(url)
                num_rows_with_nan = df.isnull().sum(axis=1).gt(0).sum()
                print("Number of rows containing NaN values:", num_rows_with_nan)

                df = df.drop(self.ignore_features, axis="columns")

                imputer = SimpleImputer(strategy='most_frequent')

                # df = df.dropna(axis=1, how='all')  # Drop columns that contain all NaN values

                # print(df.shape)
                # df = df.dropna()                
                df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


                if df.isnull().values.any() or df.empty:
                    print("Nan in synthetic")
                    exit(1)
                # Save the data to the specified file path as JSON
                df.to_json(json_file_path, orient='records', lines=True)
            else:
                # JSON file already exists locally, load it directly
                df = pd.read_json(json_file_path, orient='records', lines=True)

            df.rename(columns={'gender': 'sex'}, inplace=True)


            self.dtype_map = {
                "age": "int",
                "workclass": "category",
                "fnlwgt": "int", 
                "education": "category",
                "educational-num": "category",
                "marital-status": "category",
                "occupation": "category", 
                "relationship": "category", 
                "race": "category", 
                "sex": "category", 
                "capital-gain": "int", 
                "capital-loss": "int",
                "hours-per-week": "int", 
                "native-country": "category", 
                "Class-label": "category"
            }
            if len(self.ignore_features)>0:
                for rem_feat in self.ignore_features:
                    self.dtype_map.pop(rem_feat)
            # self.dtype_map = {}
            # for column, dtype in df.dtypes.items():
            #     if pd.api.types.is_numeric_dtype(dtype):
            #         self.dtype_map[column] = "int"
            #     else:
            #         self.dtype_map[column] = "category"

            self.column_names = list(self.dtype_map.keys())
            # Extract categorical column names


        elif dataset_name == "compas":
            self.ignore_features += ["id", "age_cat", "priors_count.1", "violent_recid"]
            url = "https://raw.githubusercontent.com/tailequy/fairness_dataset/main/experiments/data/compas-scores-two-years_clean.csv"
            target = "two_year_recid"

            # Define the file path where you want to save the data
            folder_path = "datasets/{}".format(dataset_name)

            if len(self.ignore_features) > 0:
                dataset_name += "_" + "_".join(self.ignore_features)

            json_file_path = os.path.join(folder_path, "{}.json".format(dataset_name))

            # Check if the folder exists, if not, create it
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Check if the JSON file exists locally
            if not os.path.isfile(json_file_path):
                # File does not exist locally, download the CSV dataset
                df = pd.read_csv(url)
                num_rows_with_nan = df.isnull().sum(axis=1).gt(0).sum()
                print("Number of rows containing NaN values:", num_rows_with_nan)

                df = df.drop(self.ignore_features, axis="columns")

                imputer = SimpleImputer(strategy='most_frequent')

                # df = df.dropna(axis=1, how='all')  # Drop columns that contain all NaN values

                # print(df.shape)
                # df = df.dropna()                
                df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


                if df.isnull().values.any() or df.empty:
                    print("Nan in synthetic")
                    exit(1)
                # Save the data to the specified file path as JSON
                df.to_json(json_file_path, orient='records', lines=True)
            else:
                # JSON file already exists locally, load it directly
                df = pd.read_json(json_file_path, orient='records', lines=True)

            self.dtype_map = {
                'id': 'int', 
                'name': 'category', 
                'first': 'category', 
                'last': 'category', 
                'compas_screening_date': 'category', 
                'sex': 'category', 
                'dob': 'category', 
                'age': 'int', 
                'age_cat': 'category', 
                'race': 'category', 
                'juv_fel_count': 'int', 
                'decile_score': 'int', 
                'juv_misd_count': 'int', 
                'juv_other_count': 'int', 
                'priors_count': 'int', 
                'days_b_screening_arrest': 'int', 
                'c_jail_in': 'category', 
                'c_jail_out': 'category', 
                'c_case_number': 'category', 
                'c_offense_date': 'category', 
                'c_arrest_date': 'category', 
                'c_days_from_compas': 'int', 
                'c_charge_degree': 'category', 
                'c_charge_desc': 'category', 
                'is_recid': 'category', 
                'r_case_number': 'category', 
                'r_charge_degree': 'category', 
                'r_days_from_arrest': 'int', 
                'r_offense_date': 'category', 
                'r_charge_desc': 'category', 
                'r_jail_in': 'category', 
                'r_jail_out': 'category', 
                'violent_recid': 'int', 
                'is_violent_recid': 'category', 
                'vr_case_number': 'category', 
                'vr_charge_degree': 'category', 
                'vr_offense_date': 'category', 
                'vr_charge_desc': 'category', 
                'type_of_assessment': 'category', 
                'decile_score.1': 'int', 
                'score_text': 'category', 
                'screening_date': 'category', 
                'v_type_of_assessment': 'category', 
                'v_decile_score': 'int', 
                'v_score_text': 'category', 
                'v_screening_date': 'category', 
                'in_custody': 'category', 
                'out_custody': 'category', 
                'priors_count.1': 'int', 
                'start': 'int', 
                'end': 'int', 
                'event': 'int', 
                'two_year_recid': 'category'
            }

            if len(self.ignore_features)>0:
                for rem_feat in self.ignore_features:
                    self.dtype_map.pop(rem_feat)
            # self.dtype_map = {}
            # for column, dtype in df.dtypes.items():
            #     if pd.api.types.is_numeric_dtype(dtype):
            #         self.dtype_map[column] = "int"
            #     else:
            #         self.dtype_map[column] = "category"

            self.column_names = list(self.dtype_map.keys())
            # Extract categorical column names


        elif dataset_name == "dutch":
            url = "https://raw.githubusercontent.com/tailequy/fairness_dataset/main/experiments/data/dutch.csv"
            target = "occupation"

            # Define the file path where you want to save the data
            folder_path = "datasets/{}".format(dataset_name)

            if len(self.ignore_features) > 0:
                dataset_name += "_" + "_".join(self.ignore_features)

            json_file_path = os.path.join(folder_path, "{}.json".format(dataset_name))

            # Check if the folder exists, if not, create it
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Check if the JSON file exists locally
            if not os.path.isfile(json_file_path):
                # File does not exist locally, download the CSV dataset
                df = pd.read_csv(url)
                num_rows_with_nan = df.isnull().sum(axis=1).gt(0).sum()
                print("Number of rows containing NaN values:", num_rows_with_nan)

                df = df.drop(self.ignore_features, axis="columns")

                imputer = SimpleImputer(strategy='most_frequent')

                # df = df.dropna(axis=1, how='all')  # Drop columns that contain all NaN values

                # print(df.shape)
                # df = df.dropna()                
                df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


                if df.isnull().values.any() or df.empty:
                    print("Nan in synthetic")
                    exit(1)

                # df["dummy"] = 0.0
                # Save the data to the specified file path as JSON
                df.to_json(json_file_path, orient='records', lines=True)
            else:
                # JSON file already exists locally, load it directly
                df = pd.read_json(json_file_path, orient='records', lines=True)

            self.dtype_map = {
                'sex': 'category', 
                'age': 'category', 
                'household_position': 'category', 
                'household_size': 'category', 
                'prev_residence_place': 'category', 
                'citizenship': 'category', 
                'country_birth': 'category', 
                'edu_level': 'category', 
                'economic_status': 'category', 
                'cur_eco_activity': 'category', 
                'marital_status': 'category', 
                'occupation': 'category', 
                "dummy": 'int'
            }

            if len(self.ignore_features)>0:
                for rem_feat in self.ignore_features:
                    self.dtype_map.pop(rem_feat)
            # self.dtype_map = {}
            # for column, dtype in df.dtypes.items():
            #     if pd.api.types.is_numeric_dtype(dtype):
            #         self.dtype_map[column] = "int"
            #     else:
            #         self.dtype_map[column] = "category"

            self.column_names = list(self.dtype_map.keys())
            # Extract categorical column names


        elif dataset_name == "german":
            url = "https://raw.githubusercontent.com/tailequy/fairness_dataset/main/experiments/data/german_data_credit.csv"

            target = "class-label"

            # Define the file path where you want to save the data
            folder_path = "datasets/{}".format(dataset_name)

            if len(self.ignore_features) > 0:
                dataset_name += "_" + "_".join(self.ignore_features)

            json_file_path = os.path.join(folder_path, "{}.json".format(dataset_name))

            # Check if the folder exists, if not, create it
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Check if the JSON file exists locally
            if not os.path.isfile(json_file_path):
                # File does not exist locally, download the CSV dataset
                df = pd.read_csv(url)
                num_rows_with_nan = df.isnull().sum(axis=1).gt(0).sum()
                print("Number of rows containing NaN values:", num_rows_with_nan)

                df = df.drop(self.ignore_features, axis="columns")

                imputer = SimpleImputer(strategy='most_frequent')

                # df = df.dropna(axis=1, how='all')  # Drop columns that contain all NaN values

                # print(df.shape)
                # df = df.dropna()                
                df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


                if df.isnull().values.any() or df.empty:
                    print("Nan in synthetic")
                    exit(1)
                # Save the data to the specified file path as JSON
                df.to_json(json_file_path, orient='records', lines=True)
            else:
                # JSON file already exists locally, load it directly
                df = pd.read_json(json_file_path, orient='records', lines=True)


            self.dtype_map = {
                'checking-account': 'category', 
                'duration': 'int', 
                'purpose': 'category',
                'credit-history': 'category', 
                'credit-amount': 'int', 
                'savings-account': 'category', 
                'employment-since': 'category', 
                'installment-rate': 'int', 
                'other-debtors': 'category', 
                'residence-since': 'int', 
                'property': 'category', 
                'age': 'int', 
                'other-installment': 'category', 
                'housing': 'category', 
                'existing-credits': 'int', 
                'job': 'category', 
                'numner-people-provide-maintenance-for': 'int', 
                'telephone': 'category', 
                'foreign-worker': 'category', 
                'sex': 'category', 
                'marital-status': 'category', 
                'class-label': 'category', 
            }

            if len(self.ignore_features)>0:
                for rem_feat in self.ignore_features:
                    self.dtype_map.pop(rem_feat)
            # self.dtype_map = {}
            # for column, dtype in df.dtypes.items():
            #     if pd.api.types.is_numeric_dtype(dtype):
            #         self.dtype_map[column] = "int"
            #     else:
            #         self.dtype_map[column] = "category"

            self.column_names = list(self.dtype_map.keys())
            # Extract categorical column names

        elif dataset_name == "credit":
            url = "https://raw.githubusercontent.com/tailequy/fairness_dataset/main/experiments/data/credit-card-clients.csv"

            target = "default payment"

            self.dtype_map = {
                "limit_bal": "int",
                "sex": "category",
                "education": "category",
                "marriage": "category",
                "age": "int",
                "pay_0": "category",
                "pay_2": "category",
                "pay_3": "category",
                "pay_4": "category",
                "pay_5": "category",
                "pay_6": "category",
                "bill_amt1": "int",
                "bill_amt2": "int",
                "bill_amt3": "int",
                "bill_amt4": "int",
                "bill_amt5": "int",
                "bill_amt6": "int",
                "pay_amt1": "int",
                "pay_amt2": "int",
                "pay_amt3": "int",
                "pay_amt4": "int",
                "pay_amt5": "int",
                "pay_amt6": "int",
                "default payment": "category"
            }

            if len(self.ignore_features)>0:
                for rem_feat in self.ignore_features:
                    self.dtype_map.pop(rem_feat)
            # self.dtype_map = {}
            # for column, dtype in df.dtypes.items():
            #     if pd.api.types.is_numeric_dtype(dtype):
            #         self.dtype_map[column] = "int"
            #     else:
            #         self.dtype_map[column] = "category"

            self.column_names = list(self.dtype_map.keys())
            # Extract categorical column names
            categorical_cols = [col for col, dtype in self.dtype_map.items() if dtype == "category"]




            # Define the file path where you want to save the data
            folder_path = "datasets/{}".format(dataset_name)

            if len(self.ignore_features) > 0:
                dataset_name += "_" + "_".join(self.ignore_features)

            if dataset_name == "credit":
                json_file_path = os.path.join(folder_path, "{}_transformed_positive_categorical.json".format(dataset_name))
            else:
                json_file_path = os.path.join(folder_path, "{}.json".format(dataset_name))

            # Check if the folder exists, if not, create it
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Check if the JSON file exists locally
            if not os.path.isfile(json_file_path):
                # File does not exist locally, download the CSV dataset
                df = pd.read_csv(url)
                df = df.rename(columns=lambda x: x.lower())

                num_rows_with_nan = df.isnull().sum(axis=1).gt(0).sum()
                print("Number of rows containing NaN values:", num_rows_with_nan)

                df = df.drop(self.ignore_features, axis="columns")

                # imputer = SimpleImputer(strategy='most_frequent')

                # # df = df.dropna(axis=1, how='all')  # Drop columns that contain all NaN values

                # # print(df.shape)
                # # df = df.dropna()                
                # df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


                if df.isnull().values.any() or df.empty:
                    print("Nan in synthetic")
                    exit(1)

                categorical_transformer_lgbm = Pipeline(steps=[
                    ('ordinal', PositiveOrdinalEncoder(categorical_cols))
                    ])

                transformations_positive_cat = ColumnTransformer(
                    transformers=[
                        ('all', categorical_transformer_lgbm, df.columns),
                    ])
                transformed_positive_vals = transformations_positive_cat.fit_transform(df)

                df = pd.DataFrame(transformed_positive_vals, columns=df.columns)

                df.to_json(json_file_path, orient='records', lines=True)
                # Save the data to the specified file path as JSON
                # df.to_json(json_file_path, orient='records', lines=True)
            else:
                # JSON file already exists locally, load it directly
                df = pd.read_json(json_file_path, orient='records', lines=True)
            

        else:
            print("Dataset not supported")
            exit(1)

        self.original_mappings = {}
        for column, dtype in self.dtype_map.items():
            if dtype == "category":
                self.original_mappings[column] = dict(enumerate(df[column].astype("category").cat.categories))

        self.reverse_mappings = {col: {v: k for k, v in self.original_mappings[col].items()} for col in self.original_mappings}



        df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).astype("category")

        # Display the first few rows of the dataset
        self.original_dataframe = df.copy()
        self.original_dataframe_encoded = self.encode(df).copy()

        self.target = target

        df_dummy_drop = df.copy()
        df_dummy_drop = df_dummy_drop.drop(columns=[target, self.protected_attribute])

        self.categorical_input_cols = [col for col in df_dummy_drop.columns if self.dtype_map[col] == "category"]

        self.continuous_input_cols = [col for col in df_dummy_drop if col not in self.categorical_input_cols]

        self.categorical_input_col_locations = [df_dummy_drop.columns.get_loc(col) for col in self.categorical_input_cols]


        categorical_count, numerical_count = sum(1 for dtype in self.dtype_map.values() if dtype == "category"), sum(1 for dtype in self.dtype_map.values() if dtype in ["int", "float"])
        print("Dataset {} has {} categorical and {} numerical columns.".format(dataset_name, self.categorical_input_cols, self.continuous_input_cols))
        return self




    def get_synthesizer_method(self, name, dataframe, random_state=42):
        if name=="cart":
            synthesizer = Synthpop(seed=random_state)
            arguments = {"dtypes": {col: dtype for col, dtype in self.dtype_map.items() if col in dataframe.columns}}
        elif name=="smote":
            cat_cols = [col for col in self.categorical_input_cols if col in dataframe.columns]
            synthesizer = SMOTENC_GENERATIVE(categorical_features=cat_cols, random_state=random_state)
            arguments = {}
        elif name=="tvae":
            cat_cols = [col for col in self.categorical_input_cols if col in dataframe.columns]
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(dataframe)
            synthesizer = TVAESynthesizer(metadata)
            arguments = {}
        elif name=="gaussian_copula":
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(dataframe)
            synthesizer = GaussianCopulaSynthesizer(metadata)    
            arguments = {}
        elif name=="ctgan":
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(dataframe)
            synthesizer = CTGANSynthesizer(metadata)    
            arguments = {}  
        else:
            print(name, "method not supported")
            
        return synthesizer, arguments

    def train_synthesizer(self, name, dataframe, y=[], encode=True, random_state=42):

        synthesizer, arguments = self.get_synthesizer_method(name, dataframe, random_state=random_state)

        if encode:
            dataframe = self.encode(dataframe)
        else:
            dataframe[dataframe.select_dtypes(['category']).columns] = dataframe.select_dtypes(['category']).astype("object")

        if synthesizer:
            synthesizer.fit(dataframe, **arguments)

        return synthesizer

    def generate_data(self, synthesizer, num=100, name="", decode=True, dataframe=[], random_state=42):
        if name=="tvae" or name=="ctgan" or name=="gaussian_copula":
            # synthesizer._set_random_state(random_state)
            synthetic_data = synthesizer.sample(int(num))
        else:
            synthetic_data = synthesizer.generate(int(num))

        if decode:
            synthetic_data = self.decode(synthetic_data)
        
        #sanity check
        if synthetic_data.isnull().values.any():
            print("Nan in synthetic")
            exit(1)
            return

        return synthetic_data

    def split_population(self, dataframe, protected_attributes=["sex"]):
        # Split the DataFrame based on the value pairs of protected attributes
        split_dfs = {}
        split_sizes = {}
        largest_split_size = 0

        for attr_values, indices in dataframe.groupby(protected_attributes).groups.items():
            split_df = dataframe.loc[indices]
            split_dfs[attr_values] = split_df
            split_size = len(split_df)
            split_sizes[attr_values] = split_size
            if split_size > largest_split_size:
                largest_split_size = split_size
                largest_split_attr_values = attr_values

        # Calculate the number of data each smaller split needs to add
        additional_data = {}
        for attr_values, split_size in split_sizes.items():
            if attr_values != largest_split_attr_values:
                additional_data[attr_values] = largest_split_size - split_size

        return split_dfs, additional_data



