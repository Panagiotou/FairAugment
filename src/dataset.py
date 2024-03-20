import os
import pandas as pd
from synthpop import Synthpop
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
from .our_smote import SMOTENC_GENERATIVE
import numpy as np
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

from src.mdgmm.utils_methods import encode, get_var_metadata, post_process, transform_df_to_json, generate_plots
from src.mdgmm.init_params import dim_reduce_init
from src.mdgmm.data_preprocessing import compute_nj
from src.mdgmm.MIAMI import MIAMI
from scipy.spatial.distance import pdist, squareform


class Dataset():
    def __init__(self, dataset_name, binary_features=True, ignore_features=["fnlwgt"]):
        self.binary_features = binary_features
        self.ignore_features = ignore_features
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
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            column_names_original = [
                "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                "hours-per-week", "native-country", "income"
            ]
            self.dtype_map = {
                "age": "int",
                "workclass": "category",
                "fnlwgt": "int", 
                "education": "category",
                "marital-status": "category",
                "occupation": "category", 
                "relationship": "category", 
                "race": "category", 
                "sex": "category", 
                "capital-gain": "int", 
                "capital-loss": "int",
                "hours-per-week": "int", 
                "native-country": "category", 
                "income": "category"
            }
            if len(self.ignore_features)>0:
                for rem_feat in self.ignore_features:
                    self.dtype_map.pop(rem_feat)

            self.column_names = list(self.dtype_map.keys())
            # Extract categorical column names
            self.categorical_cols = [col for col, dtype in self.dtype_map.items() if dtype == "category"]
            self.categorical_input_cols = self.categorical_cols.copy()

            self.continuous_input_cols = [col for col in self.column_names if col not in self.categorical_cols]

            self.categorical_input_cols.remove("income")

            self.binary_cols = ["native-country", "race"]

        else:
            print("Dataset not supported")
            exit(1)


        # Define the file path where you want to save the data
        folder_path = "datasets/{}".format(dataset_name)

        if self.binary_features:
            dataset_name += "_binary"
        if len(self.ignore_features)>0:
            dataset_name += "_" + "_".join(self.ignore_features)


        json_file_path = os.path.join(folder_path, "{}.json".format(dataset_name))

        # Check if the folder exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Check if the JSON file exists locally
        if not os.path.isfile(json_file_path):
            # File does not exist locally, download the CSV dataset

            df = pd.read_csv(url, names=column_names_original, na_values=' ?', skipinitialspace=True)
            # Drop columns that are not in self.column_names
            df = df.drop(columns=[col for col in df.columns if col not in self.column_names])

            if self.binary_features:
                df['race'] = df['race'].apply(lambda x: x.strip() if x.strip() == 'White' else 'Other')

                # Preprocess "native-country" feature
                df['native-country'] = df['native-country'].apply(lambda x: x.strip() if x.strip() == 'United-States' else 'Other')

            # Save the data to the specified file path as JSON
            df.to_json(json_file_path, orient='records', lines=True)
        else:
            # JSON file already exists locally, load it directly
            df = pd.read_json(json_file_path, orient='records', lines=True)


    
        # Applying get_dummies for categorical columns
        # df_transformed = self.encode(df)

        self.original_mappings = {}
        for column, dtype in self.dtype_map.items():
            if dtype == "category":
                self.original_mappings[column] = dict(enumerate(df[column].astype("category").cat.categories))

        self.reverse_mappings = {col: {v: k for k, v in self.original_mappings[col].items()} for col in self.original_mappings}



        df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).astype("category")

        # Display the first few rows of the dataset
        self.original_dataframe = df.copy()
        self.original_dataframe_encoded = self.encode(df).copy()
        return self

    def get_synthesizer_method(self, name, dataframe, random_state=42):
        if name=="cart":
            synthesizer = Synthpop(seed=random_state)
            arguments = {"dtypes": {col: dtype for col, dtype in self.dtype_map.items() if col in dataframe.columns}}
        if name=="smote":
            cat_cols = [col for col in self.categorical_input_cols if col in dataframe.columns]
            synthesizer = SMOTENC_GENERATIVE(categorical_features=cat_cols, random_state=random_state)
            arguments = {}
        if name=="tvae":
            cat_cols = [col for col in self.categorical_input_cols if col in dataframe.columns]
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(dataframe)
            synthesizer = TVAESynthesizer(metadata)
            arguments = {}
        if name=="mdgmm":
            return None, None

            
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
        if name=="tvae":
            synthesizer._set_random_state(random_state)
            synthetic_data = synthesizer.sample(int(num))
        elif name=="mdgmm":
            dataframe = self.encode(dataframe)
            var_distrib, var_transform_only, le_dict = get_var_metadata(dataframe, self.binary_cols)
            p = dataframe.shape[1]
            dtypes_dict_famd = {'continuous': float, 'categorical': str, 'ordinal': str,\
              'bernoulli': str, 'binomial': str}
            
            dtype = {dataframe.columns[j]: dtypes_dict_famd[var_transform_only[j]] for j in range(p)}
            dataframe_famd = dataframe.astype(dtype, copy=True)

            nj, nj_bin, nj_ord, nj_categ = compute_nj(dataframe, var_distrib)

            n_clusters = 3
            r = np.array([2, 1])
            k = [n_clusters]
            # Feature category (cf)

            
            


            authorized_ranges = None

            init, transformed_famd_data, famd  = dim_reduce_init(dataframe_famd, n_clusters, k, r, nj, var_distrib, seed = 2023, use_light_famd=True)
            distances = pdist(transformed_famd_data)

            dm = squareform(distances)

            discrete_features = self.categorical_input_cols
            target_nb_pseudo_obs= int(num)
            eps = 1E-10
            it = 1000
            maxstep = 1000
            seed = 2023

            out = MIAMI(dataframe, n_clusters, r, k, init, var_distrib, nj, le_dict, authorized_ranges, discrete_features, dtype, target_nb_pseudo_obs=target_nb_pseudo_obs, it=it,\
                eps=eps, maxstep=maxstep, seed=seed, perform_selec = True, dm = dm, max_patience = 0)

            synthetic_data = pd.DataFrame(out['y_all'], columns = dataframe.columns) 
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



