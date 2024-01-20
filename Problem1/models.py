from data import DataLoader
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout,Conv2DTranspose,BatchNormalization,Activation,Reshape
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
import numpy as np
import wandb
from wandb.keras import WandbCallback
import os
os.environ["WANDB_API_KEY"] = "cdd71ae76c07e41586c6cbf5b1c72c12200201e4"

class VAE(tf.keras.Model):
    def __init__(self, latent_dim,dropout_rate = 0):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.encoder_build(dropout_rate)
        self.latent_space_encoder = self.latent_space_build()
        self.decoder_first_layer = self.decoder_first_layer_build()
        self.decoder = self.decoder_build(dropout_rate)
        
    def encoder_build(self,dropout_rate):
        encoder = tf.keras.models.Sequential([
            Conv2D(filters=32, kernel_size=(3, 3),strides=2, activation='relu',padding='same', input_shape=(96, 96, 3)),
            BatchNormalization(),
            Dropout(dropout_rate),
            Conv2D(filters=64, kernel_size=(3, 3),strides=2, activation='relu',padding='same'),
            BatchNormalization(),
            Dropout(dropout_rate),
            Conv2D(filters=128, kernel_size=(5, 5),strides=1, activation='relu',padding='same'),
            BatchNormalization(),
            Conv2D(filters=1, kernel_size=(1, 1),strides=1, activation='relu',padding='same'),
        ])
        #encoder.summary()
        return encoder
    
    def latent_space_build(self):
        latent_space = tf.keras.models.Sequential([
            Flatten(input_shape = (24,24,1)),   
            Dense(2 * self.latent_dim), 
            
        ])
        #latent_space.summary()
        return latent_space

    # the first layer after latent space
    def decoder_first_layer_build(self):
        decoder = tf.keras.models.Sequential([
            Dense(units=24*24*1, activation='relu', input_shape=(self.latent_dim,)),
            Reshape(target_shape=(24, 24,1)), # resize to "image format"
            Conv2DTranspose(filters=1, kernel_size=(1, 1),strides=1, activation='relu',padding='same'),
        ])
        #decoder.summary()
        return decoder
    
    # the second to last layer of the decoder
    def decoder_build(self,dropout_rate):
        decoder = tf.keras.models.Sequential([
            Conv2DTranspose(filters=128, kernel_size=(5, 5),strides=1, activation='relu',padding='same', input_shape=(24, 24, 1)),
            BatchNormalization(),
            Dropout(dropout_rate),
            Conv2DTranspose(filters=64, kernel_size=(3, 3),strides=2, activation='relu',padding='same'),
            BatchNormalization(),
            Dropout(dropout_rate),
            Conv2DTranspose(filters=32, kernel_size=(3, 3),strides=2, activation='relu',padding='same'),
            BatchNormalization(),
            Dropout(dropout_rate),
            Conv2DTranspose(filters=3, kernel_size=1,strides=1, activation='relu'),
        ])
        #decoder.summary()
        return decoder
    
    
    def encode(self, x):
        return self.encoder(x)
    
    def latent_space(self,x):
        param = self.latent_space_encoder(x)
        return tf.split(param, num_or_size_splits=2, axis=1) # mean, logvar    
    
    def decode_first_layer(self,z):
        return self.decoder_first_layer(z)
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean #sigma= sqrt(exp(logvar))
    
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return tf.sigmoid(self.decode(self.decode_first_layer(eps)))
   
class TrainVAE():
    def __init__(self,latent_dim = 24*24):
        self.data = DataLoader()
        self.model = VAE(latent_dim)
        
        # init the sweep config for search
        self.sweep_config = {
                'method': 'grid',
                'metric': {
                    'name': 'val_loss',
                    'goal': 'minimize'
                },
                'early_terminate':{
                    'type': 'hyperband',
                    'min_iter': 5
                },  
                'parameters': {
                    'learning_rate':{
                        'values': [0.001, 0.0005,0.0001]
                        },
                    'dropout_rate': {
                        'values': [0.5]
                    },
                    'optimizer': {
                        'values': ['adam']
                    },
                    "latent_space": {
                        'values': [24, 24*24, 24*24*10]
                        }
                }   
            }
        self.sweep_id = wandb.sweep(sweep=self.sweep_config,project="VAE")
        
    # logarithm of the probability according to a normal distribution
    def log_normal_pdf(self,sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        vals = -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)
        return tf.reduce_sum(vals, axis=raxis)

    # loss function 
    def compute_loss(self,model, x):
        encoded = model.encode(x)
        mean, logvar = model.latent_space(encoded)
        z = model.reparameterize(mean, logvar)    
        z_prime = model.decode_first_layer(z)
        x_logit = model.decode(z_prime)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, 
                                                            labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])    
        logpz = self.log_normal_pdf(z, 0., 0.)    
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    @tf.function
    def train_step(self, model, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        return gradients
    
    # trains the model. If called direcly it trains for config_defaults else use train() for sweep
    def run(self, epochs=1):
        train_dataset, _, val_dataset = self.data.create_batches_image()
        config_defaults = {
            "learning_rate": 0.001,
            "dropout_rate": 0.5,
            "latent_space": 24,
            "optimizer": 'adam'
        }
        wandb.init(config=config_defaults,project="AE")
        wandb.config.epochs = epochs
        self.model = VAE(wandb.config.latent_space,wandb.config.dropout_rate)
        
        optimizer = wandb.config.optimizer
        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate)
        if optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=wandb.config.learning_rate, momentum=0.9)
        
        checkpoint_path = "Models_training_VAE_/" + wandb.config.optimizer+"_"+str(wandb.config.learning_rate)+"_"+str(wandb.config.dropout_rate)+"_"+str(wandb.config.latent_space)+ "/model.ckpt"
        for epoch in range(epochs):
            self.best_val_loss = float('-inf')
            for train_x in train_dataset:                
                gradients = self.train_step(self.model, train_x)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            # get training loss
            train_loss = tf.keras.metrics.Mean()
            for train_x in train_dataset:
                train_loss(self.compute_loss(self.model, train_x))
            variational_lower_bound_train = -train_loss.result()
                
            # get validation loss
            loss = tf.keras.metrics.Mean()
            for val_x in val_dataset:
                loss(self.compute_loss(self.model, val_x))
            variational_lower_bound = -loss.result()

            wandb.log({'epoch': epoch + 1, 'val_loss': variational_lower_bound, 'train_loss': variational_lower_bound_train})

            print(f'Epoch: {epoch}, Validation set variational lower bound: {variational_lower_bound}')

            # Save model
            if variational_lower_bound > self.best_val_loss:
                self.best_val_loss = variational_lower_bound
                self.model.save_weights(checkpoint_path)
    
    # create a sweep         
    def train(self, epochs = 1):
        wandb.agent(self.sweep_id, function=lambda: self.run(epochs))
       
    #  get latent space of one image from training    
    def latent_space(self):
        train_dataset,_,_ = self.data.create_batches_image()
        for train_x in train_dataset:
                return self.model.encode(train_x)
            
    # encode and decode an image
    def encode_decode(self, image):
        encode = self.model.encoder.predict(image, verbose = 0)
        decode = self.model.decoder.predict(encode, verbose = 0)
        return decode
        
    # load model
    def load(self,checkpoint_path, optimizer = "adam", lr = 0.1, dropout_rate = 0, latent_space = 24*24):
        self.model = VAE(latent_space,dropout_rate)
        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        self.model.load_weights(checkpoint_path)
        return self.model
    
    # plot images and reconstructions of images    
    def plot(self,n):
        train_dataset,_,_ = self.data.create_batches_image()
        plt.figure(figsize=(20, 4))
        for i, images in enumerate(train_dataset.take(n)):  # Take 1 batch for visualization
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(images[i].numpy())
            plt.title("original")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            ax = plt.subplot(2, n, i + 1 + n)
            encoded_image = self.model.encode(images)
            mean, logvar = self.model.latent_space(encoded_image)
            
            z = self.model.reparameterize(mean, logvar)
            predictions = self.model.sample(z)
            
            plt.imshow(predictions[i][:, :, 0])
            plt.title("reconstructed")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

class AE(tf.keras.Model):
    def __init__(self, dropout_rate = 0, regularization = True):
        super(AE, self).__init__()
        self.encoder = self.encoder_build(dropout_rate,regularization)
        self.decoder = self.decoder_build(dropout_rate,regularization)
        
    def encoder_build(self, dropout_rate,regularization):
        if regularization:
            encoder = tf.keras.models.Sequential([
                Conv2D(filters=32, kernel_size=(3, 3),strides=2, activation='relu',padding='same', input_shape=(96, 96, 3)),
                BatchNormalization(),
                Dropout(dropout_rate),
                Conv2D(filters=64, kernel_size=(3, 3),strides=2, activation='relu',padding='same'),
                BatchNormalization(),
                Dropout(dropout_rate),
                Conv2D(filters=128, kernel_size=(5, 5),strides=1, activation='relu',padding='same'),
                BatchNormalization(),
                Conv2D(filters=1, kernel_size=(1, 1),strides=1, activation='relu',padding='same'),
            ])
        else:
            encoder = tf.keras.models.Sequential([
                Conv2D(filters=32, kernel_size=(3, 3),strides=2, activation='relu',padding='same', input_shape=(96, 96, 3)),
                Conv2D(filters=64, kernel_size=(3, 3),strides=2, activation='relu',padding='same'),
                Conv2D(filters=128, kernel_size=(5, 5),strides=1, activation='relu',padding='same'),
                Conv2D(filters=1, kernel_size=(1, 1),strides=1, activation='relu',padding='same'),
            ])
        #encoder.summary()
        return encoder
    
    def decoder_build(self,dropout_rate,regularization):
        if regularization:
            decoder = tf.keras.models.Sequential([
                Conv2DTranspose(filters=128, kernel_size=(5, 5),strides=1, activation='relu',padding='same', input_shape=(24, 24, 1)),
                BatchNormalization(),
                Dropout(dropout_rate),
                Conv2DTranspose(filters=64, kernel_size=(3, 3),strides=2, activation='relu',padding='same'),
                BatchNormalization(),
                Dropout(dropout_rate),
                Conv2DTranspose(filters=32, kernel_size=(3, 3),strides=2, activation='relu',padding='same'),
                BatchNormalization(),
                Dropout(dropout_rate),
                Conv2DTranspose(filters=3, kernel_size=1,strides=1, activation='relu'),
            ])
        else:
            decoder = tf.keras.models.Sequential([
                Conv2DTranspose(filters=128, kernel_size=(5, 5),strides=1, activation='relu',padding='same', input_shape=(24, 24, 1)),
                Conv2DTranspose(filters=64, kernel_size=(3, 3),strides=2, activation='relu',padding='same'),
                Conv2DTranspose(filters=32, kernel_size=(3, 3),strides=2, activation='relu',padding='same'),
                Conv2DTranspose(filters=3, kernel_size=1,strides=1, activation='relu'),
            ])
        #decoder.summary()
        return decoder
    
    def encode(self, x):
        return self.encoder(x)
    
class TrainAE():
    def __init__(self):
        self.data = DataLoader()
        self.model = AE()
        
        # init the sweep config for search
        self.sweep_config = {
                'method': 'grid',
                'metric': {
                    'name': 'val_loss',
                    'goal': 'minimize'
                },
                'early_terminate':{
                    'type': 'hyperband',
                    'min_iter': 5
                },  
                'parameters': {
                    'learning_rate':{
                        'values': [0.001, 0.0005,0.0001]
                        },
                    'dropout_rate': {
                        'values': [0.2, 0.5, 0.7]
                    },
                    'optimizer': {
                        'values': ['adam', 'sgd']
                    },
                    'regularization':{
                        'values': [True]
                    }
                }   
            }
        self.sweep_id = wandb.sweep(sweep=self.sweep_config,project="AE")
        
    # trains the model. If called direcly it trains for config_defaults else use train() for sweep
    def run(self, epochs = 1):
        config_defaults = {
            "learning_rate": 0.001,
            "dropout_rate": 0.5,
            "optimizer": 'adam',
            "regularization": True
        }
        wandb.init(config=config_defaults,project="AE_regularization")
        wandb.config.epochs = epochs
        
        train_dataset,_,val_dataset = self.data.create_batches_image_image()
        
        self.model = AE(dropout_rate=wandb.config.dropout_rate, regularization=wandb.config.regularization,)
        optimizer = wandb.config.optimizer
        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate)
        if optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=wandb.config.learning_rate, momentum=0.9)
        autoencoder = tf.keras.models.Sequential([self.model.encoder, self.model.decoder], name='autoencoder')
        autoencoder.compile(loss='mse', optimizer=optimizer)
        
        checkpoint_path = "Models_training_AE_/" + wandb.config.optimizer+"_"+str(wandb.config.learning_rate)+"_"+str(wandb.config.dropout_rate)+ "/model.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)
        
        autoencoder.fit(train_dataset, validation_data=val_dataset, epochs=wandb.config.epochs,
                         callbacks=[cp_callback, WandbCallback(input_type="image")],)
        wandb.finish()

    # create a sweep
    def train(self, epochs = 1):
        wandb.agent(self.sweep_id, function=lambda: self.run(epochs))    
        
    def latent_space(self):
        train_dataset,_,_ = self.data.create_batches_image()
        for train_x in train_dataset:
            return self.model.encoder(train_x)
    
    def load(self,checkpoint_path, optimizer = "adam", lr = 0.1, dropout_rate = 0):
        self.model = AE(dropout_rate)
        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        model = tf.keras.models.Sequential([self.model.encoder, self.model.decoder], name='autoencoder')
        model.compile(loss='mse', optimizer=optimizer)
        model.load_weights(checkpoint_path)
        return model

    # plot images and their encoding              
    def plot_latent_space(self,n):
        train_dataset,_,_ = self.data.create_batches_image()
        plt.figure(figsize=(20, 4))
        for i, images in enumerate(train_dataset.take(n)):  # Take 1 batch for visualization
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(images[i].numpy())
            plt.title("original")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            ax = plt.subplot(2, n, i + 1 + n)
            image = self.latent_space()
            plt.imshow(image[i][:, :, 0],cmap='gray')
            plt.title("reconstructed")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
    
    def encode_decode(self, image):
        encode = self.model.encoder.predict(image, verbose = 0)
        decode = self.model.decoder.predict(encode, verbose = 0)
        return decode
    
    # plot images and reconstructions of images    
    def plot(self,n):
        train_dataset,_,_ = self.data.create_batches_image()
        plt.figure(figsize=(20, 4))
        for i, images in enumerate(train_dataset.take(n)):  # Take 1 batch for visualization
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(images[i].numpy())
            plt.title("original")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            ax = plt.subplot(2, n, i + 1 + n)
            image = self.encode_decode(images)
            plt.imshow(image[i][:, :, 0])
            plt.title("reconstructed")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        
class CNN(tf.keras.Model):
    def __init__(self,dropout_rate = 0):
        super(CNN, self).__init__()
        self.model = self.model_build(dropout_rate)
        
    def model_build(self,dropout_rate):
        input_img = Input(shape=(24,24,1))
                
        x = Conv2D(16, (3, 3), padding='valid')(input_img)
        x = BatchNormalization()(x)  # Add BatchNormalization after Conv2D
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)  # Add Dropout after Activation

        x = Conv2D(32, (3, 3), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)  # Add Dropout after Activation


        x = Conv2D(64, (3, 3), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)  # Add Dropout after Activation

        x = Conv2D(64, (3, 3), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)  # Add Dropout after Activation

        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        y = Dense(2, activation='softmax')(x)
        model = Model(inputs=input_img, outputs=y)
        return model


class TrainCNN():
    def __init__(self, image_model = "resize", image_type = "resize"):
        self.data = DataLoader()
        self.model = CNN()
        self.image_model = image_model
        self.image_type = image_type
        
        # init the sweep config for search
        self.sweep_config = {
                'method': 'random',
                'metric': {
                    'name': 'val_loss',
                    'goal': 'minimize'
                },
                'early_terminate':{
                    'type': 'hyperband',
                    'min_iter': 5
                },  
                'parameters': {
                    'learning_rate':{
                        'values': [0.1, 0.01,0.001,0.0001]
                        },
                    'dropout_rate': {
                        'values': [0.2,0.5]
                    },
                    'optimizer': {
                        'values': ['adam', 'sgd']
                    }
                }   
            }
        self.sweep_id = wandb.sweep(sweep=self.sweep_config,project="CNN_" + self.image_type)
        
    # trains the model. If called direcly it trains for config_defaults else use train() for sweep
    def run(self, epochs = 1):
        config_defaults = {
            "learning_rate": 0.001,
            "dropout_rate": 0.2,
            "optimizer": 'adam'
        }
        
        wandb.init(config=config_defaults)
        wandb.config.epochs = epochs
        
        train_dataset,_,val_dataset = self.get_data()

        optimizer = wandb.config.optimizer
        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate)
        if optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=wandb.config.learning_rate, momentum=0.9)
        self.model = CNN(wandb.config.dropout_rate)
        model = self.model.model
        model.compile(loss='mse', optimizer=optimizer,metrics=['accuracy'])
        checkpoint_path = "Models_training_CNN_/" + wandb.config.optimizer+"_"+str(wandb.config.learning_rate)+"_"+str(wandb.config.dropout_rate)+ "/model.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)
        
        model.fit(train_dataset, validation_data=val_dataset, epochs=wandb.config.epochs,
                         callbacks=[cp_callback, WandbCallback(input_type="image")],)
        self.model_compiled = model
        wandb.finish()
     
    # create sweep   
    def train(self, epochs = 1):
        wandb.agent(self.sweep_id, function=lambda: self.run(epochs))
    
    
    # converts data to the right format. If no AE or VAE is used, use the model to encode else resize the image
    def get_data(self):
        train_dataset_raw,test_dataset_raw,val_dataset_raw = self.data.create_batches_image_label()
        if self.image_type == "resize":
            def preprocess_data(image, label):
                resized_image = tf.image.resize(image, (24, 24))
                # Convert the image to grayscale (1 channel)
                grayscale_image = tf.image.rgb_to_grayscale(resized_image)
                return grayscale_image, label
            train_dataset = train_dataset_raw.map(preprocess_data)
            test_dataset = test_dataset_raw.map(preprocess_data)
            val_dataset = val_dataset_raw.map(preprocess_data)
        elif self.image_type == "AE":
            def preprocess_data_AE(image, label):
                return self.image_model.model.encode(image),label
            train_dataset = train_dataset_raw.map(preprocess_data_AE)
            test_dataset = test_dataset_raw.map(preprocess_data_AE)
            val_dataset = val_dataset_raw.map(preprocess_data_AE)
        elif self.image_type == "VAE_encode":
            def preprocess_data_VAE_encode(image, label):
                return self.image_model.model.encode(image),label
            train_dataset = train_dataset_raw.map(preprocess_data_VAE_encode)
            test_dataset = test_dataset_raw.map(preprocess_data_VAE_encode)
            val_dataset = val_dataset_raw.map(preprocess_data_VAE_encode)
        elif self.image_type == "VAE_decode":
            def preprocess_data_VAE_decode(image, label):
                encoded = self.image_model.model.encode(image)
                mean, logvar = self.image_model.model.latent_space(encoded)
                z = tf.exp(logvar * 0.5) + mean
                z_prime = self.image_model.model.decode_first_layer(z)
                return z_prime,label
            train_dataset = train_dataset_raw.map(preprocess_data_VAE_decode)
            test_dataset = test_dataset_raw.map(preprocess_data_VAE_decode)
            val_dataset = val_dataset_raw.map(preprocess_data_VAE_decode)
        else:
            print("WRONG INPUT TYPE FOR image_type")
            quit()
        return  train_dataset, test_dataset ,val_dataset 
    
    # load model
    def load(self,checkpoint_path, optimizer = "adam", lr = 0.1, dropout_rate = 0):
        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        cnn = CNN(dropout_rate)
        model = cnn.model
        model.compile(loss='mse', optimizer=optimizer,metrics=['accuracy'])
        model.load_weights(checkpoint_path)
        return model
    

class Pretrained(tf.keras.Model):
    def __init__(self,dropout_rate = 0):
        super(Pretrained, self).__init__()
        self.model = self.model_build(dropout_rate)
    def model_build(self,dropout_rate):
        input_img = Input(shape=(3, 3, 1280))
        x = Flatten()(input_img)
        x = Dense(128, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        y = Dense(2, activation='softmax')(x)
        model = Model(inputs=input_img, outputs=y)
        return model        

class TrainPretrained():
    def __init__(self):
        self.data = DataLoader()
        self.model = Pretrained()
        
        # init the sweep config for search
        self.sweep_config = {
                'method': 'random',
                'metric': {
                    'name': 'val_loss',
                    'goal': 'minimize'
                },
                'early_terminate':{
                    'type': 'hyperband',
                    'min_iter': 5
                },  
                'parameters': {
                    'learning_rate':{
                        'values': [0.01,0.01, 0.001,0.0001]
                        },
                    'dropout_rate': {
                        'values': [0.2, 0.5, 0.7]
                    },
                    'optimizer': {
                        'values': ['adam', 'sgd']
                    }
                }   
            }
        self.sweep_id = wandb.sweep(sweep=self.sweep_config,project="Pretrained")
    
    # trains the model. If called direcly it trains for config_defaults else use train() for sweep
    def run(self, epochs = 1):
        config_defaults = {
            "learning_rate": 0.001,
            "dropout_rate": 0.2,
            "optimizer": 'adam'
        }
        
        wandb.init(config=config_defaults)
        wandb.config.epochs = epochs
        
        train_dataset,_,val_dataset = self.get_data()
        optimizer = wandb.config.optimizer
        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate)
        if optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=wandb.config.learning_rate, momentum=0.9)
        self.model = Pretrained(wandb.config.dropout_rate)
        model = self.model.model
        model.compile(loss='mse', optimizer=optimizer,metrics=['accuracy'])
        checkpoint_path = "Models_training_Pretrained_/" + wandb.config.optimizer+"_"+str(wandb.config.learning_rate)+"_"+str(wandb.config.dropout_rate)+ "/model.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)
        
        model.fit(train_dataset, validation_data=val_dataset, epochs=wandb.config.epochs,
                         callbacks=[cp_callback, WandbCallback()],)
        self.model_compiled = model
        wandb.finish()
        
    def train(self, epochs = 1):
        wandb.agent(self.sweep_id, function=lambda: self.run(epochs), count = 5)
    
    
    def load(self,checkpoint_path, optimizer = "adam", lr = 0.1, dropout_rate = 0):
        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        self.model = Pretrained(dropout_rate)
        model = self.model.model
        model.compile(loss='mse', optimizer=optimizer,metrics=['accuracy'])
        model.load_weights(checkpoint_path)
    
    def run_test(self, optimizer = 'sgd'):
        _,test_dataset,_ = self.get_data()
        model = self.model.model
        model.compile(loss='mse', optimizer=optimizer,metrics=['accuracy'])
        model.evaluate(test_dataset)
 
    def get_data(self):
        train_dataset_raw, test_dataset_raw, val_dataset_raw = self.data.create_batches_image_label()

        def preprocess_data(image, label):
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(96, 96, 3),
                include_top=False,
                weights='imagenet',
            )
            base_model.trainable = False

            processed_image = base_model(image)  
            return processed_image, label

        # Create new features
        train_dataset = train_dataset_raw.map(preprocess_data)
        test_dataset = test_dataset_raw.map(preprocess_data)
        val_dataset = val_dataset_raw.map(preprocess_data)

        return train_dataset, test_dataset, val_dataset

        
