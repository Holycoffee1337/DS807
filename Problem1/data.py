import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import numpy as np
import os
BATCH_SIZE = 32
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# for dataload, manually set path or 
class DataLoader():
    def __init__(self, path='Data/patch_camelyon/'):
        self.path = path
        self.train_dataset, self.validation_dataset, self.test_dataset = self.load_data()
    
    def convert_sample_image_label(self, sample):
        image, label = sample['image'], sample['label']
        image = tf.image.convert_image_dtype(image, tf.float32, name = 'image')
        label = tf.one_hot(label, 2, dtype=tf.float32, name = 'label')
        return image, label
    
    def convert_sample_image_image(self, sample):
        image = sample['image']
        image = tf.image.convert_image_dtype(image, tf.float32, name = 'image')
        return image, image
    
    def convert_sample_image(self, sample):
        image = sample['image']
        image = tf.image.convert_image_dtype(image, tf.float32, name = 'image')
        return image

    def create_batches_image_label(self):
        train_dataset = self.train_dataset.map(self.convert_sample_image_label).batch(BATCH_SIZE)
        validation_dataset = self.validation_dataset.map(self.convert_sample_image_label).batch(BATCH_SIZE)
        test_dataset = self.test_dataset.map(self.convert_sample_image_label).batch(BATCH_SIZE)
        return train_dataset, validation_dataset, test_dataset
    
    def create_batches_image_image(self):
        train_dataset = self.train_dataset.map(self.convert_sample_image_image).batch(BATCH_SIZE)
        validation_dataset = self.validation_dataset.map(self.convert_sample_image_image).batch(BATCH_SIZE)
        test_dataset = self.test_dataset.map(self.convert_sample_image_image).batch(BATCH_SIZE)
        return train_dataset, validation_dataset, test_dataset
    
    def create_batches_image(self):
        train_dataset = self.train_dataset.map(self.convert_sample_image).batch(BATCH_SIZE)
        validation_dataset = self.validation_dataset.map(self.convert_sample_image).batch(BATCH_SIZE)
        test_dataset = self.test_dataset.map(self.convert_sample_image).batch(BATCH_SIZE)
        return train_dataset, validation_dataset, test_dataset
            
    def load_data(self):
        ds1, ds2, ds3 = tfds.load('patch_camelyon', split=['train[:1%]', 'test[:5%]', 'validation[:5%]'],
                                  data_dir=self.path,
                                  download=False,
                                  shuffle_files=True)
        return ds1, ds2, ds3

