# Import our files
from absl.testing.parameterized import parameters
from keras.src.engine.data_adapter import DataHandler
from tensorflow._api.v2 import train
from tensorflow._api.v2.random import create_rng_state
from wandb.sdk.wandb_sweep import sweep
import modelClass
import wandb
from wandb.keras import WandbCallback
import os
import nltk
from nltk.corpus import stopwords
import time
import matplotlib as plt
import warnings
warnings.filterwarnings('ignore')

# Our Files
import data_preprocessing
from data_preprocessing import DataHandler, loadDataHandler
import visualization
import shallow_learning
from shallow_learning import SVM, DT, BOOSTING, RF 

 #import shallow_learning

##### Initialize wandb #####
os.environ["WANDB_API_KEY"] = '76fdf9acb6a4a334b5b8c8f89c8a63c1c2b5135e'
project_name = 'Chris_Test' 
entity = 'marcs20'             
wandb.login()
# Set wandb to offline mode
os.environ['WANDB_MODE'] = 'dryrun'
############################


# Sweep config file, for RNN hyperparameter search
sweep_config = {
        'method': 'random',  # Can be 'grid', 'random', or 'bayes'
        'metric': {
           'name': 'val_accuracy',
           'goal': 'maximize'   
        },
        'parameters': { # output_sequence_length: Choose from plot
                       #'batch_size': {
                       #'values': [128, 224]
                # },
                # 'embedding_dimension': {
                #     'values': [200, 400, 1000]
            
            'n_epochs': {
                'values': [5]
            },

            'lr': {
                'values': [0.1, 0.01, 0.001, 0.0001]
            },
            'output_sequence_length':{
                'values': [25, 100]
            },
            'embedding_dimension':{
                'values': [200, 400]
            },

                       # embedding_dimension=200,     # 
                

                       # output_sequence_length=200,   # 75% Fractil: 90% Fractil

        }
}

# Categories we want to use for our dataset
CATEGORIES = [
              'Magazine_Subscriptions', 
              'Clothing_Shoes_and_jewelry', # BIGGEST 
              'All_Beauty',
              'Arts_Crafts_and_Sewing', 
              'AMAZON_FASHION']
 


def create_and_save_data_classes(CATEGORIES):
    '''
    This function create the data classes:
        - DATA_CLASS_WITHOUT_STOP_WORDS
        - DATA_CLASS
        - DATA_CLASS_UNI
        - DATA_CLASS_SMALL
    '''

    # Creating and saving Data Class - Without equal balnce ratings and stopwords
    DataHandler = data_preprocessing.DataHandler(CATEGORIES, class_name='DATA_CLASS_WITHOUT_STOP_WORDS')
    DataHandler.create_and_save_data()
    DataHandler.load_all_categories()
    DataHandler.saveDataHandlerClass('DATA_CLASS_WITHOUT_STOP_WORDS') 

    # Creating and saving Data Class - Without equal balnce ratings                     
    DataHandler1 = data_preprocessing.DataHandler(CATEGORIES, class_name='DATA_CLASS')   
    DataHandler1.create_and_save_data()                                                  
    DataHandler1.load_all_categories()                                                   
    DataHandler1.update_remove_stop_words()                                              
    DataHandler1.saveDataHandlerClass('DATA_CLASS')                                      

    # Creating and saving Data Class
    DataHandler2 = data_preprocessing.DataHandler(CATEGORIES, class_name='DATA_CLASS_UNI') 
    DataHandler2.create_and_save_data()                                                
    DataHandler2.load_all_categories()                                                 
    DataHandler2.update_remove_stop_words()                                            
    DataHandler2.update_balance_ratings()
    DataHandler2.saveDataHandlerClass('DATA_CLASS_UNI')                                                

    # Creating and saving small Data Class
    DataHandler3 = data_preprocessing.DataHandler(CATEGORIES[:1], class_name='DATA_CLASS_SMALL') 
    DataHandler3.create_and_save_data()                                                
    DataHandler3.load_all_categories()                                                 
    DataHandler3.update_remove_stop_words()                                            
    DataHandler3.update_balance_ratings()
    DataHandler3.saveDataHandlerClass('DATA_CLASS_SMALL')
    
    # Summarize the data for the 3 classes
    DataHandler1.summarize_data()
    DataHandler2.summarize_data()                                                      
    DataHandler3.summarize_data()

    return None


def execute_all_shallow_models(data_class):

    """
    This function is ued to execute,
    all shallow learning models,
    on specific dataclass.
    All results is saved.
    """

    print("Executing SVM Model")
    SVM(data_class)
    print("SVM Model Execution Completed")

    print("Executing RF Model")
    RF(data_class)
    print("RF Model Execution Completed")

    print("Executing Boosting Model")
    BOOSTING(data_class)
    print("Boosting Model Execution Completed")

    print("Executing DT Model")
    DT(data_class)
    print("DT Model Execution Completed")


   

# Save plot for given class
def save_plots(DataClass):
    vis = visualization.DataVisualizer(DataClass)
    vis.plot_review_ratio()
    vis.plot_avg_n_word_reveiw()
    vis.plot_common_words()
    vis.plot_n_sentence_length()

def save_all_plots():
    DataClassSmall = data_preprocessing.loadDataHandler('DATA_CLASS_SMALL') 
    DataClassUni = data_preprocessing.loadDataHandler('DATA_CLASS_UNI')     
    DataClass = data_preprocessing.loadDataHandler('DATA_CLASS')            

    save_plots(DataClassSmall)
    save_plots(DataClassUni)
    save_plots(DataClass)

def train_with_wandb():
    # Load Data Handler
    # DATA: data_preprocessing.DataHandler = data_preprocessing.loadDataHandler('DATA_CLASS')
    # DATA_UNI: data_preprocessing.DataHandler = data_preprocessing.loadDataHandler('DATA_CLASS_UNI')
    # DATA: data_preprocessing.DataHandler = data_preprocessing.loadDataHandler('DATA_CLASS')
    data_class_name = 'DATA_CLASS_SMALL_UNI'
    print(f'##### {data_class_name} #####')
    DATA_SMALL = data_preprocessing.loadDataHandler(data_class_name)
    DataClass = DATA_SMALL 

    # Initialize wandb for this training rucn
    run = wandb.init(project='1_TRY')  # Corrected project name
    config = run.config

    # Initialize and build the model using hyperparameters indb
    model = modelClass.Build_Models(DataClass,
                                    epochs_n=config.n_epochs)
                                    # batch_size=config.batch_size)
    model.build_model()
    model.train()
    # model.evaluate()

    # Finish the wandb run after training is complete
    run.finish()



if __name__ == "__main__":
    start_time = time.time()
    
    ##############################################    
    ##### Create and save the 2 data Classes #####
    #### Delete training, before call create #####
    ######## Remember to Delete files ############
    ##############################################

    create_and_save_data_classes(CATEGORIES)
    
    ##############################################
    ############## LOAD DATAHANDLER ##############
    ##############################################

    DataClassSmall = data_preprocessing.loadDataHandler('DATA_CLASS_SMALL')
    DataClassUni = data_preprocessing.loadDataHandler('DATA_CLASS_UNI')
    DataClass = data_preprocessing.loadDataHandler('DATA_CLASS')

    ##############################################
    ############## VISUALIZE #####################
    ##############################################

    save_all_plots() # Save all plots for the 3 DataClasses

    ##############################################
    ############## EXECUTE  ######################
    ########### SHALLOW MODELS ###################
    ##############################################

    execute_all_shallow_models(DataClassSmall)
    execute_all_shallow_models(DataClassUni)
    execute_all_shallow_models(DataClass)

    ##############################################
    ########### Creating Sweep (RNN) ############# 
    ##############################################

    # sweep_id = wandb.sweep(sweep_config, project=project_name)
    # wandb.agent(sweep_id, train_with_wandb, count=5)

     
    # Print total time for execution 
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")
