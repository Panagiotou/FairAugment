import os
import pandas as pd
from synthpop import Synthpop
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import matplotlib.pyplot as plt

class Dataset():
    def __init__(self, dataset_name, binary_features=True):
        self.binary_features = binary_features
        dataframe = self.load_dataset(dataset_name)

    def encode(self, df, keep_dtypes=False):
        df_encoded = df.copy()
        for column in df_encoded.columns:
            dtype = self.dtype_map[column]
            if dtype == "category":
                df_encoded[column] = df_encoded[column].map(self.reverse_mappings[column]).astype("category")
        if keep_dtypes:
            df_encoded = df_encoded.astype(df.dtypes)
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

            self.column_names = list(self.dtype_map.keys())
            # Extract categorical column names
            self.categorical_cols = [col for col, dtype in self.dtype_map.items() if dtype == "category"]
            self.categorical_input_cols = self.categorical_cols.copy()
            self.categorical_input_cols.remove("income")
        else:
            print("Dataset not supported")
            exit(1)


        # Define the file path where you want to save the data
        folder_path = "datasets/{}".format(dataset_name)

        if self.binary_features:
            dataset_name += "_binary"
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

        # Display the first few rows of the dataset
        self.original_dataframe = df.copy()
        self.original_dataframe_encoded = self.encode(df).copy()
        return self

    def train_synthesizer(self, dataframe, encode=True):
        synthesizer = Synthpop()
        dtypes = {col: dtype for col, dtype in self.dtype_map.items() if col in dataframe.columns}
        if encode:
            dataframe_transformed = self.encode(dataframe)
            synthesizer.fit(dataframe_transformed, dtypes)
        else:
            synthesizer.fit(dataframe, dtypes)

        return synthesizer

    def generate_data(self, synthesizer, num=100):
        synthetic_data = synthesizer.generate(int(num))
        synthetic_data_decoded = self.decode(synthetic_data)
        return synthetic_data_decoded

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



