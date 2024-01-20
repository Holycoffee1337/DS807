# RNN Model
import pandas as pd
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
import wandb
from wandb.keras import WandbCallback
from sklearn.metrics import classification_report


# Build RNN Model
class Build_Models():
    def __init__(self, DataHandler,
                 category_name = None,
                 model_name = 'Model_name',
                 max_tokens=1000,               # 1000 
                 output_sequence_length=200,   # 75% Fractil: 90% Fractil
                 pad_to_max_tokens=True,
                 batch_size=64,
                 embedding_dimension=200,     # 
                 epochs_n=10):
    
        self.filename =  model_name
        self.category_name = category_name
        self.model = None
        self.encoder = None

    
        # Hyperparameters                                    
        self.epochs_n = epochs_n                             
        self.max_tokens = max_tokens                         # 75% Fractil: 90% Fractil
        self.output_sequence_length = output_sequence_length #  
        self.pad_to_max_tokens = pad_to_max_tokens           
        self.embedding_dimension = embedding_dimension       
        self.batch_size = batch_size                         
        self.epochs_n = epochs_n                             

        DataHandler.summarize_data()

        # Data 
        self.DataHandler = DataHandler
        # If no category is specified, use the combined training data
        self.data_train = self.DataHandler.get_combined_train_data()
        self.data_val = self.DataHandler.get_combined_val_data()

        # Encoder
        self._adapt_encoder()
        
        # Ds data
        self.ds_train = self._preprocess_data(self.data_train['text'], self.data_train['overall']) 
        self.ds_val = self._preprocess_data(self.data_val['text'], self.data_val['overall']) 


    def build_model(self):
        early_stopping = EarlyStopping(monitor='accuracy',  # or 'val_accuracy' depending on your preference
                                       patience=10,  # Number of epochs with no improvement after which training will be stopped
                                       restore_best_weights=True)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(input_dim=len(self.encoder.get_vocabulary()), 
                                      output_dim=wandb.config.embedding_dimension,  # self.embedding_dimension,
                                      input_length=wandb.config.output_sequence_length,     # self.output_sequence_length,
                                      trainable=True, # Embeddings Trainable (Defulat in Tensor) 
                                      name="embedding"), 
            # Dropout(0.5),
            LayerNormalization(axis=-1), # Normalize the Embedding 
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))),
            Dropout(0.5),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))),
            Dropout(0.5),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))),
            Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.5),
            tf.keras.layers.Dense(5, activation='softmax')
        ])


            

        
        # Note legacy Adam optimizer used instead of 'adam'
        self.model.compile(optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=wandb.config.lr),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])


    def _adapt_encoder(self):
        self.encoder = tf.keras.layers.TextVectorization(max_tokens=self.max_tokens, 
                                            output_sequence_length=wandb.config.output_sequence_length,
                                            pad_to_max_tokens=self.pad_to_max_tokens)
        text_ds = tf.data.Dataset.from_tensor_slices(self.data_train['text']).batch(self.batch_size)
        self.encoder.adapt(text_ds)


    def _preprocess_data(self, text_data, label_data):
        # Create the full dataset with text and labels
        ds = tf.data.Dataset.from_tensor_slices((text_data, label_data)).batch(self.batch_size) #  Maybe wandb.
        # Apply TextVectorization to the text data in the dataset
        ds = ds.map(lambda x, y: (self.encoder(x), y))
        # Configure the dataset for performance
        AUTOTUNE = tf.data.AUTOTUNE 
        ds = ds.cache().prefetch(buffer_size=AUTOTUNE)
        return ds


    # def train(self, early_stopping_patience=3):
    #     early_stopping = EarlyStopping(monitor='accuracy',
    #                                    patience=early_stopping_patience,
    #                                    restore_best_weights=True)

    #     # Use WandB callback for logging
    #     wandb_callback = wandb.keras.WandbCallback()

    #     model_train = self.model.fit(self.ds_train, 
    #                    epochs=wandb.config.n_epochs,  #self.epochs_n, 
    #                    validation_data=self.ds_val,
    #                    callbacks=[early_stopping, wandb_callback], 
    #                    verbose = 1)

    def train(self, early_stopping_patience=10):
        early_stopping = EarlyStopping(monitor='val_accuracy',
                                       patience=early_stopping_patience,
                                       restore_best_weights=True)

        # Use WandB callback for logging
        wandb_callback = wandb.keras.WandbCallback()

        history = self.model.fit(self.ds_train, 
                                 epochs=wandb.config.n_epochs,  # Make sure this is set correctly in your wandb.config
                                 validation_data=self.ds_val,
                                 callbacks=[early_stopping, wandb_callback],
                                 verbose=1)

        # After training, evaluate and create the report on the validation set
        # First, get the true labels and predictions
        y_true = []
        y_pred = []
        for inputs, labels in self.ds_val:
            predictions = self.model.predict(inputs)
            # Convert predictions to discrete labels if they are probabilities
            predicted_labels = np.argmax(predictions, axis=1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted_labels)
        
        # Ensure true labels and predictions are in the correct format
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Generate the classification report
        report = classification_report(y_true, y_pred, digits=2)
        print(report)

        # Optionally, log the classification report to WandB
        # wandb.log({'Classification Report': report})

        # Return the history object containing information about the training process
        return history



    def save_model(self):
        self.model.save(self.filename)


    def evaluate(self):
        test_data = self.DataHandler.get_test_data()
        # Evaluate the model on the test set
        test_loss, test_accuracy = self.model.evaluate(test_data)


        header = "Model Evaluation"
        print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}")
        print(f'Test Loss: {test_loss}')
        print(f'Test Accuracy: {test_accuracy}')




if __name__ == "__main__":
    quit()

