'''
File: data_preprocessing.py
Purpose: provides DataHandler class,
that handle all the data.
'''
from pandas.core.internals.construction import get_objs_combined_axis
import tensorflow as tf
import pandas as pd
import requests
import gzip
import io
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import pickle
import os
import nltk
from nltk.corpus import stopwords
warnings.simplefilter('ignore')
import glob
import json

class DataHandler:

    """
    DataHandler is a class to handle data preprocessing,
    for Amazon Reviews data.

    Precondition:
        Path: './Data/Locally/'
        Path: './Data/Training/'
        Local json files for given categories in path '/Data/Locally'

    Parameters:
        Categoies: List of categories you want data for 
pass 
    """

    def __init__(self, categories,
                 current_category = None,
                 class_name = None,
                 chunk_size = 300,
                 base_path = './Data/Training/',
                 json_path = './Data/Locally/'):
        self.base_path = base_path # Path to folder
        self.json_path = json_path # Path to Json files
        self.chunk_size = chunk_size

        # Create folder setup for project
        self._mkdir_folder_setup()

        # Just to keep track, on last category loaded
        self.current_category = current_category
        
        # The test data
        self.test_category_name = None
        self.test_data = None
        self._load_test_data()
        
        # Combined data for all categories
        self.combined_train_data = None
        self.combined_val_data = None
        self.combined_data = None
        
        # Dictionary, to hold val, train and entire dataframe for each category
        self.categories = categories
        self.data_sets = {category: {'train': None, 'val': None, 'df': None} for category in categories}

        # When loading 
        self.class_name = class_name  # attribute to store the name of the data class

    def _mkdir_folder_setup(self):
        '''
        Created needed folder setup folder for user. 
        
            Precondition: Json files in ./Data/Locally/
            To replicate data, reduce json files to 0.1gb for,
                - Clothing_Shoes_and_jewelry_5
                - Arts_Crafts_and_Sewing_5)
            Done, to reduce workload for hardware.
        '''

        header = "Creating folder setup"
        print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}")
    
        folders = [
            "./Data/",
            "./Data/Training/",
            "./Data/Locally/",  # Put JSON Files into this folder
            "./Data/SmallJsonFiles/",
            "./RESULTS/",
            "./RESULTS/DATA_CLASSES/",
            "./RESULTS/PLOTS/",
            "./RESULTS/PLOTS/REVIEW_RATIO/",
            "./RESULTS/PLOTS/COMMON_WORDS/",
            "./RESULTS/PLOTS/WORD_COUNT_DISTRIBUTION/",
            "./RESULTS/PLOTS/AVG_WORDS/",
            "./RESULTS/RNN/",
            "./RESULTS/SHALLOW/",
            "./RESULTS/SHALLOW/SAVED_MODELS_SHALLOW/",
            "./RESULTS/SHALLOW/CLASSIFICATION_REPORT/",
            "./RESULTS/DATA_SUMMARY/"
        ]
    
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"Created folder: {folder}")
            else:
                print(f"Folder already exists: {folder}")
    

    # This method create and save the preprocessed,
    def create_and_save_data(self): 

        csv_files = glob.glob(os.path.join(self.base_path, '*.csv'))
        
        # Delete all current csv files, 
        # To prevent, appending on current ones
        print(f'\nDelete .csv files in folder: {self.base_path}, before creating files. So it does not append to current files.')
        for file_path in csv_files:
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except OSError as e:
                print(f"Error: {e.strerror} - {file_path}")

        header = "Creating and Saving Data"
        print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}")
        print(f"Destination for saved files: {self.base_path}")

        valid_categories = df_read_save_json(self.categories, self.base_path)
        self.categories = valid_categories

     
    def update_balance_ratings(self):

        """
        This method sets the data, with equal amount of
        reviews for each rating and update the class. 
        """

        header = "Update to uniform ratings"
        print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}")


        for category in self.data_sets:
            train_data =  self.data_sets[category]['train']
            val_data =  self.data_sets[category]['val'] # REMEMBER TO DELETE THIS -> ONLY FOR TEST
            self.data_sets[category]['train'] = equal_rating_groups(train_data)
            # This is veyry stupid -> # self.data_sets[category]['val'] = equal_rating_groups(val_data)
            if len(self.data_sets[category]['val']) > len(self.data_sets[category]['train']):
                # Truncate the validation data
                train_size = len(self.data_sets[category]['train'])
                self.data_sets[category]['val'] = self.data_sets[category]['val'].sample(n=train_size, random_state=42).reset_index(drop=True)



        self._combine_data()
        df_train = self.combined_train_data
        df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
        self.combined_train_data = df_train
        
        self.combined_data = df_train + self.combined_val_data 


    def update_remove_stop_words(self):

        """
        This method, remove all the stop words,
        for all reviews, and updates the class.
        """

        header = "Remove stop words from data"
        print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}")

        # Remove stop words from all category in list
        for category in self.data_sets.keys():
            # Apply remove_stop_words to train, validation, and dataframe data of each category
            self.data_sets[category]['train'] = remove_stop_words(self.data_sets[category]['train'])
            self.data_sets[category]['val'] = remove_stop_words(self.data_sets[category]['val'])
            self.data_sets[category]['df'] = remove_stop_words(self.data_sets[category]['df'])
        
        # Remove stop words from combined data
        self.combined_train_data = remove_stop_words(self.combined_train_data)
        self.combined_val_data = remove_stop_words(self.combined_val_data)
        self.combined_data = remove_stop_words(self.combined_data)

        # Remove stop words from test data
        self.test_data = remove_stop_words(self.test_data)
     

    
    def load_category(self, category):

        '''
        This method, load data for the given category,
        and update the class, with the data, for
        given category.

        Parameters:
            category (str): Takes the company name

        '''

        header = 'Load category into class'
        print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}")
        print(f'Loaded category: {category}')

        if category not in self.categories:
            raise ValueError(f"Category '{category}' not in category list.")

        X_train = pd.read_csv(f"{self.base_path}{category}_X_train.csv")
        X_val = pd.read_csv(f"{self.base_path}{category}_X_val.csv")

        self._preprocess_data(X_train)
        self._preprocess_data(X_val)

        self.data_sets[category]['train'] = X_train
        self.data_sets[category]['val'] = X_val
        self.data_sets[category]['df'] = pd.concat([X_train, X_val]) 

    

    def load_all_categories(self):

        '''
        This method, take the category list,
        and load all the data in the class, for each category.
        train, val- and df. (Preprocessed)
        '''

        header = "Load all categories into class"
        print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}")


        for category in self.categories:
            print(f'Loaded: {category}')

            def process_chunks(file_path):
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                    chunks.append(chunk)
                return pd.concat(chunks, axis=0) # axis = 0??

            # Load training data
            X_train = process_chunks(f"{self.base_path}{category}_X_train.csv")
            self._preprocess_data(X_train)  # Preprocess the concatenated DataFrame

            # Load validation data
            X_val = process_chunks(f"{self.base_path}{category}_X_val.csv")
            self._preprocess_data(X_val)  # Preprocess the concatenated DataFrame

            # Load df data
            # print(f"{self.json_path}{category}_5.json")
            full_df_path = f"{self.json_path}{category}_5.json"
            full_df = json_to_df(full_df_path)
            full_df = dataFormatting(full_df)

            # Store in the data_sets dictionary
            self.data_sets[category] = {'train': X_train, 'val': X_val, 'df': full_df}
        
        print(f'Loaded: Combined categories into one dataframe')
        self._combine_data()


    def _load_test_data(self):

        """
        This method, load the test data 'Luxury_Beauty,
        into to the class. This is done, automatically' 
        """
        
        category = 'Luxury_Beauty'
        full_df_path = f"{self.json_path}{category}_5.json"
        full_df = json_to_df(full_df_path)
        full_df = dataFormatting(full_df)
        self.test_data = full_df
        self.test_category_name = category




    def _combine_data(self):

        """
        This method, takes all categories data,
        and combine it, in one dataframe.
        This method are called automatically,
        when loading_all_categories, are called.
        """

        all_train_data = []
        all_val_data = []

        all_train_data = []
        all_val_data = []
    
        for category in self.data_sets:
            if self.data_sets[category]['train'] is not None:
                train_data = self.data_sets[category]['train']
                all_train_data.append(self.data_sets[category]['train'])
    
            if self.data_sets[category]['val'] is not None:
                val_data = self.data_sets[category]['val'] 
                all_val_data.append(self.data_sets[category]['val'])

            self.data_sets[category]['df'] = pd.concat([train_data, val_data])


        # Concatenate all training and validation data
        combined_train_data = pd.concat(all_train_data, ignore_index=True) if all_train_data else None
        combined_val_data = pd.concat(all_val_data, ignore_index=True) if all_val_data else None
    
        # Shuffle all the data
        combined_train_data_shuffled = combined_train_data.sample(frac=1, random_state=42).reset_index(drop=True)
        combined_val_data_shuffled = combined_val_data.sample(frac=1, random_state=42).reset_index(drop=True)
        combined_data = pd.concat([combined_train_data_shuffled, combined_val_data_shuffled], ignore_index=True)
    
        self.combined_train_data = combined_train_data_shuffled 
        self.combined_val_data = combined_val_data_shuffled
        self.combined_data = combined_data

    def summarize_data(self):
        '''
        This method, summarize the the data for the class.
        It also save, the summarize, as an .csv. 
        '''

        header = f"Data Summary for {self.class_name}: Saved as csv in path './RESULTS/SUMMARY/data_summary_{self.class_name}.csv'" # if self.class_name else "Data Summary"
        print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}\n")

        summary_data = []

        for category, datasets in self.data_sets.items():
            summary = {
                'Category': category,
                'Train Set Size': len(datasets['train']),
                'Validation Set Size': len(datasets['val']),
                'Combined Set Size': len(datasets['df']),
                'Train Distribution': datasets['train']['overall'].value_counts().sort_index().to_dict(),
                'Validation Distribution': datasets['val']['overall'].value_counts().sort_index().to_dict(),
            }
            summary_data.append(summary)

        summary_df = pd.DataFrame(summary_data)

        max_length_train = max(summary_df['Train Distribution'].astype(str).apply(len))
        summary_df['Train Distribution'] = summary_df['Train Distribution'].astype(str).apply(lambda x: x.ljust(max_length_train + 4))

        max_length_val = max(summary_df['Validation Distribution'].astype(str).apply(len))
        summary_df['Validation Distribution'] = summary_df['Validation Distribution'].astype(str).apply(lambda x: x.ljust(max_length_val))

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        print(summary_df.to_string(index=False))

        print('')
        # Assuming self.test_data is a DataFrame with test data
        test_data_summary = {
            'Size': len(self.test_data),
            'Distribution': self.test_data['overall'].value_counts().sort_index().to_dict()
        }
        test_data_summary_df = pd.DataFrame([test_data_summary])

        print("Test Data Summary (Luxury_Beauty):")
        print(test_data_summary_df.to_string(index=False))

        print("\n" + "=" * len(header))

        # Save the summary DataFrame to a CSV file
        test_data_summary_df.to_csv(f'./RESULTS/DATA_SUMMARY/test_data_summary_{self.test_category_name}.csv', index=False)
        summary_df.to_csv(f'./RESULTS/DATA_SUMMARY/data_summary_{self.class_name}.csv', index=False)
        
        return summary_df


    def _preprocess_data(self, df):

        df['overall'] = df['overall'] - 1  # Adjusting the 'overall' rating
        df['text'] = df['text'].fillna('').astype(str)


    def get_data(self, category, data_type):
        self.current_category = category
        return self.data_sets[category][data_type]

    def get_current_cateogry(self):
        return self.current_category

    def get_list_of_categories(self):
        return self.categories

    def get_combined_train_data(self):
        return self.combined_train_data

    def get_combined_val_data(self):
        return self.combined_val_data

    def get_combined_data(self):
        return self.combined_data
        
    def get_class_name(self):
        print(self.class_name)
        return self.class_name
    
    def get_test_data(self):
        return self.test_data

    # Save Datahanler
    def saveDataHandlerClass(self, file_name):
        folder_path = './RESULTS/DATA_CLASSES/'
        file_path = os.path.join(folder_path, file_name) 

        message = f"File saved as: {file_name}"
        border_length = len(message)
        header = "Saving DataHandler Class"
        print(f"\n{'=' * border_length}\n{header}\n{message}")
        print('=' * border_length)

        with open(file_path, 'wb') as file:
            pickle.dump(self, file)


# Balance the dataset by an equal numbers of rows  
def equal_rating_groups(df):
    min_rating_group = df['overall'].value_counts().min() 
    return df.groupby('overall').apply(lambda x: x.sample(min_rating_group)).reset_index(drop=True)


# Remover stop words on df
def remove_stop_words(df):
    stop_words = set(stopwords.words('english')) 
    filtered_text = []
    for text in df['text']:
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_text.append(" ".join(filtered_words))
    df['text'] = filtered_text
    return df 
 

# Used to gzip files from url.
def read_gzipped_json_from_url(url):
    # Send a HTTP request to the URL
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Use gzip to decompress the content
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
            # Read the JSON lines file and convert to a DataFrame
            df = pd.read_json(gz, lines=True)
        return df
    else:
        print(f"Failed to retrieve data: status code {response.status_code}")
        return None


# Concatenate summary with review text, to one column
def dataFormatting(df):
    # Concatenating summary and reviewText
    df['text'] = df['summary'] + ' ' + df['reviewText']
    df['text'].fillna('', inplace=True)
    columns =  ['overall', 'text']
    new_df = df[columns]
    return new_df 


# Split df into train- and validation data
def split_train_val(df):
    df_X_train, df_X_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['overall'])
    return df_X_train, df_X_val


# Split df into train- and validation data
def split_train_val2(df):
    y = df['overall']
    df_small = df[['text']]
    df_X_train, df_X_val, df_y_train, df_y_val = train_test_split(df_small, y, test_size=0.2, random_state=42, stratify=y)
    return df_X_train, df_X_val, df_y_train, df_y_val


# Save the splitted train and validation data as csv files 
def saveLocallyTraining(df, name, path_save):

    # Split the data into training and validation sets
    df_X_train, df_X_val = split_train_val(df) 

    # Reset index for all splits
    df_X_train = df_X_train.reset_index(drop=True)
    df_X_val = df_X_val.reset_index(drop=True)

    # Define the mode for opening the file, 'a' for append and 'w' for write
    mode_X_train = 'a' if os.path.exists(path_save + name + '_X_train.csv') else 'w'
    mode_X_val = 'a' if os.path.exists(path_save + name + '_X_val.csv') else 'w'

    # Write the data to CSV files, append if file exists, otherwise write
    df_X_train.to_csv(path_save + name + '_X_train.csv', mode=mode_X_train, index=False, header=not os.path.exists(path_save + name + '_X_train.csv'))
    df_X_val.to_csv(path_save + name + '_X_val.csv', mode=mode_X_val, index=False, header=not os.path.exists(path_save + name + '_X_val.csv'))


# Read json file as df
# Save the the df with splits
def df_read_save_json(l, path_save):
    chunk_size = 1000 # 10000
    path = './Data/Locally/'
    valid_categories = []
    for item in l:
        json_path = path + item + '_5.json'
        if not os.path.exists(json_path):
             print('')
             print(f"File {json_path} does not exist. Skipping...")
             print('Download the json file (category) manually, and put in ./Data/Locally/')
             print('')
             continue
        print(f'Downloaded: {item}')
        valid_categories.append(item)
        json_reader = pd.read_json(json_path, lines = True, chunksize = chunk_size)
        for chunk in json_reader:
            chunk = dataFormatting(chunk)
            saveLocallyTraining(chunk, item, path_save)
    print('Categories do now have train and val data saved')
    return valid_categories


# Read json as df
def json_to_df(category_path):
    return pd.read_json(category_path, lines = True)

# Load data_handler class 
def loadDataHandler(class_name):
    path = './RESULTS/DATA_CLASSES/'
    class_path = path + class_name
    with open(class_path, 'rb') as input:
        data_handler = pickle.load(input)
        data_handler.get_class_name = data_handler.get_class_name()
        return data_handler


# Reduce json file, into given size
def reduce_json_files(max_size_gb=0.050):
    original_file_path = './Data/Locally/Clothing_Shoes_and_jewelry_5.json'
    new_file_path = './Data/SmallJsonFiles/Clothing_Shoes_and_jewelry_5.json'

    # original_file_path = './Data/Locally/Arts_Crafts_and_Sewing_5.json'   
    # new_file_path = './Data/SmallJsonFiles/Arts_Crafts_and_Sewing_5.json' 

    file_size_bytes = os.path.getsize(original_file_path)
    file_size_gb = file_size_bytes / (1024 * 1024 * 1024)  # Convert bytes to gigabytes

    if file_size_gb >= max_size_gb:
        with open(original_file_path, 'r', encoding='utf-8') as original_file, \
             open(new_file_path, 'w', encoding='utf-8') as new_file:
            
            bytes_written = 0
            max_bytes = max_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes

            for line in original_file:
                # Check if adding the next line exceeds the maximum size
                if bytes_written + len(line.encode('utf-8')) > max_bytes:
                    break
                new_file.write(line)
                bytes_written += len(line.encode('utf-8'))
        print(f"Created a smaller file: {new_file_path}")
    else:
        print(f"The file is smaller than {max_size_gb}GB. No new file created.")

# reduce_json_files()
