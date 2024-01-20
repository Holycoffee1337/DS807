'''
File: visualization.py
Purpose: Provide functions for visualization, 
of the give data.
'''
import pandas as pd
from pandas._libs.hashtable import value_count
import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.layers import Embedding, LSTM, Dense, Dropout, LayerNormalization
from sklearn.model_selection import train_test_split
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
# Own Files imported
import data_preprocessing 


SAVE_PATH = './RESULTS/PLOTS/'


class DataVisualizer:
    def __init__(self, DataHandler):
        self.DataHandler = DataHandler

    # Plots for list of categories
    def plot_avg_n_word_reveiw(self):
        """
        The function plot the average number of words per review,
        and summary (we have added together) for each category in the list.

        Parameters:
           DataHandler: It takes the DataHandler Class 

        Returns:
            None

        Execution: 
            Show bar plot of the average words for each review + summary.
            For all categories.
        """

        cat_avg_n_word_review(self.DataHandler)
        return cat_avg_n_word_review(self.DataHandler)

    def plot_review_ratio(self):
        """
        Calculate the procent for each key, given the total values overall.

        Parameters:
            DataHandler: It takes the DataHandler Class 

        Returns:
            None

        Execute:
            Show plot of categories and there % ratings

        """
        reviewRatio(self.DataHandler)
        return reviewRatio(self.DataHandler)


    # Plots for specific Category
    def plot_common_words(self, category=None):
        """
        Analyzes the text data in the 'text' column of the provided DataFrame and
        prints the top 10 most common words along with their frequencies.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the 'text' column to analyze.

        Returns:
            None

        Execution: 
            Show plot
        """
        if category != None:
            df = self.DataHandler.get_data(category, 'df')
        else:
            df = self.DataHandler.get_combined_data()
            category = self.DataHandler.get_class_name + '_COMBINED' 
        common_words(df, category)
        return common_words(df, category)

    def plot_n_sentence_length(self, category_name=None):
        """
        Find the length of all sentences for the category chosen, 
        and make a histogram over it. To visualize, how long
        each sentence is.
    
        Parameters:
            cateogry_name: The name for the category, you want to plot 
    
        Returns:
            None
    
        Execution: 
            Show histogram plot
        """
        
        if category_name != None:
            df = self.DataHandler.get_data(category_name, 'df')
        else:
            df = self.DataHandler.get_combined_train_data()
            category_name = self.DataHandler.get_class_name + '_COMBINED'
        word_count_distribution(df, category_name)
        return word_count_distribution(df, category_name)


def count_words(text):
    if pd.isna(text):  # Check for NaN values
        return 0
    words = str(text).split()  # Convert to string and split
    return len(words)



def word_count_distribution(df, name):

    plot_path = 'WORD_COUNT_DISTRIBUTION/'
    path = SAVE_PATH + plot_path + name + '_word_distribution'



    df['Word_Count'] = df['text'].apply(count_words)
    # print(df['Word_Count'].mean())

    percentiles = df['Word_Count'].describe(percentiles=[0.25, 0.5, 0.75])

    word_count_dict = {}
    for count in df['Word_Count']:
        if count in word_count_dict:
            word_count_dict[count] += 1
        else:
            word_count_dict[count] = 1
    
    # Convert the dictionary items to lists for plotting
    word_counts = list(word_count_dict.keys())
    row_counts = list(word_count_dict.values())

    # Create fig
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(8, 8))

    table_data = [['25%', f"{percentiles['25%']:.2f}"],
                ['50%', f"{percentiles['50%']:.2f}"],
                ['75%', f"{percentiles['75%']:.2f}"]]
    table = ax2.table(cellText=table_data, loc='center', 
                     colLabels=['Percentile', 'Value'], cellLoc='center', colColours=['#f0f0f0']*2)


    ax2.axis('off')
    # Plotting the bar chart
    ax1.set_xlabel('Word Count')
    ax1.set_ylabel('Number of Rows')
    ax1.set_title(name + ': Word Count Distribution in Rows')
    ax1.bar(word_counts, row_counts, color='blue')
    # plt.show()
    plt.savefig(path)
    plt.close()
    


def cat_avg_n_word_review(DataHandler):

    plot_path = 'AVG_WORDS/'
    class_name = DataHandler.get_class_name
    path = SAVE_PATH + plot_path + class_name + '_AVG_WORDS'

    dict_list = []
    categories = DataHandler.get_list_of_categories()
    for category in categories:
        df = DataHandler.get_data(category, 'df')  # Assuming get_data method exists
        df['Word_Count'] = df['text'].apply(count_words)
        dict_list.append({category: df['Word_Count'].mean()})

    categories = [list(d.keys())[0] for d in dict_list]
    avg_words = [list(d.values())[0] for d in dict_list]

    plt.bar(categories, avg_words, color='blue')
    plt.xlabel('Categories')
    plt.ylabel('Average Words')
    plt.title('Average Number of Words for Different Categories')
    # plt.show()
    plt.savefig(path)
    plt.close()



    
    
def calcProcent(dic): 

    """
    Calculate the procent for each key, given the total values overall.

    Parameters:
        Dicitionary: The Dictionary containing the ratings as keys, and total as values 

    Returns:
        array: Where each position correlates to the key - 1

    """

    procent_list = []
    total_reviews = dic[0] + dic[1] + dic[2] + dic[3] + dic[4]
    i = 0
    total_procent = 0
    while i < 5:
        procent = (100 * (dic[i] / total_reviews))
        procent_list.append(procent)
        i = i + 1
        total_procent = procent + total_procent
        
    # print('Total Procent')
    # print(total_procent)
    return procent_list




def reviewRatio(DataHandler):
    """
    Calculate the procent for each key, given the total values overall.

    Parameters:
       l (list): List containing names of categories 
       path: path to the locally folder containing .json files for each category 

    Returns:
        None

    Execute:
        Show plot of categories and there % ratings

    """
    class_name = DataHandler.get_class_name
    plot_path = 'REVIEW_RATIO/'
    path = SAVE_PATH + plot_path + class_name + '_review_ratio'

    dict_list = []
    categories = DataHandler.get_list_of_categories()
    # print(categories)
    for category in categories:
        df = DataHandler.get_data(category, 'df')
        dict_list.append(df['overall'].value_counts().to_dict())

        
    One = []
    Two = []
    Three = []
    Four = [] 
    Five = []
    
    for item in dict_list:
        array_procent = calcProcent(item)
        ii = 0 
        while ii < len(array_procent):
            if ii == 0:
                One.append(array_procent[ii])
            if ii == 1:
                Two.append(array_procent[ii])
            if ii == 2:
                Three.append(array_procent[ii])
            if ii == 3:
                Four.append(array_procent[ii])
            if ii == 4:
                Five.append(array_procent[ii])
            ii = ii + 1


    # set width of bar 
    barWidth = 0.15
    fig = plt.subplots(figsize =(12, 8)) 
     

    # Set position of bar on X axis 
    br1 = np.arange(len(One)) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 
    br4 = [x + barWidth for x in br3] 
    br5 = [x + barWidth for x in br4] 
     
    # Make the plot
    plt.bar(br1, One, color ='r', width = barWidth, 
            edgecolor ='grey', label ='One') 
    plt.bar(br2, Two, color ='g', width = barWidth, 
            edgecolor ='grey', label ='Two') 
    plt.bar(br3, Three, color ='b', width = barWidth, 
            edgecolor ='grey', label ='Three') 
    plt.bar(br4, Four, color ='black', width = barWidth, 
            edgecolor ='grey', label ='Four') 
    plt.bar(br5, Five, color ='purple', width = barWidth, 
            edgecolor ='grey', label ='Five') 
     

    # Adding Xticks 
    plt.xlabel('Categories', fontweight ='bold', fontsize = 15) 
    plt.ylabel('N Reviews', fontweight ='bold', fontsize = 15) 
    #plt.xticks([r + barWidth for r in range(len(One))], 
    #        DataHandler.get_list_of_categories)


    plt.xticks([r + barWidth for r in range(len(One))], DataHandler.get_list_of_categories())
     
    plt.legend()
    # plt.show()
    plt.savefig(path)
    plt.close()

    return fig 




def common_words(df_format, category_name):
    """
    Analyzes the text data in the 'text' column of the provided DataFrame and
    prints the top 10 most common words along with their frequencies.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the 'text' column to analyze.

    Returns:
        None

    Execution: 
        Show plot
    """
    
    plot_path = 'COMMON_WORDS/'
    path = SAVE_PATH + plot_path + category_name + '_common_words'




    # Handle NaN values by replacing them with an empty string
    df_format['text'] = df_format['text'].replace(np.nan, '', regex=True)
    # Use CountVectorizer to tokenize and count word frequencies
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_format['text'])
    
    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum the occurrences of each word across all sentences
    word_frequencies = Counter(dict(zip(feature_names, X.sum(axis=0).A1)))
    # print(word_frequencies['positive'])
    # Remove specific keys
    # word_frequencies.pop('positive', None)
    # word_frequencies.pop('negative', None)
    
    # Display the most common words and their frequencies
    most_common_words = word_frequencies.most_common(10)
    most_common_words = most_common_words[0:10]
    #for word, frequency in most_common_words:
        # print(f"{word}: {frequency}")
    
    # Plot a bar chart of word frequencies
    plt.bar(*zip(*most_common_words))
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(category_name + ': ' + 'Top 10 Most Common Words')
    plt.savefig(path)
    plt.close()
    # plt.show()


def save_visualize(DataClass):
    path_folder = './RESULTS/PLOTS/'
    class_name = DataClass.get_class_name
    DataVisualizer = DataVisualizer(DataClass) 
    DataVisualizer.plot_avg_n_word_reveiw()
 
