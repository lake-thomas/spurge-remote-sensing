 '''
 Script to run the CNN LSTM model using a time-series of satellite imagery
 Model expects inputs of (12 dates, 128px, 128px, 4 bands)
 
 '''


# packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.nice(20)
import sys
import datetime
import time
import random
from glob import glob
import numpy as np
import skimage
import skimage.io as skio
from skimage.io import imread, imshow, imsave
from skimage.transform import rotate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from functools import partial, update_wrapper
from itertools import product
from IPython.display import HTML, display, clear_output
import tifffile
from tabulate import tabulate
from math import ceil
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score, jaccard_score, fbeta_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, TimeDistributed, Bidirectional, ConvLSTM2D, BatchNormalization, Conv3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import *
import segmentation_models as sm

SM_FRAMEWORK = tf.keras
sm.set_framework('tf.keras')
from segmentation_models import Unet
from segmentation_models.utils import set_trainable
from segmentation_models.losses import CategoricalCELoss


# set variables
n_features = 4
image_size = 128
n_classes = 7
batch_size = 4
timesteps = 12

class_weights = [0, 0.0066, 0.0562, 0.0182, 0.0447, 0.0569, 0.8142] #proportional to pixel frequency and sums to ~1 (sum to ~1 so little to no effect on learning rate)

for i in range(len(class_weights)): 
    print(("%.17f" % class_weights[i]).rstrip('0').rstrip('.'))


# load pretrained cnn model
dummy_model = load_model(r'F:/.../models/mymodel.h5')
    
# define new model without last activation layer
base_model = Model(dummy_model.inputs, dummy_model.layers[-2].output)

base_model.set_weights(dummy_model.get_weights())

#freeze/unfreeze all pretrained cnn layers
base_model.trainable = True

# define LSTM model
inp = Input(shape=(timesteps, image_size, image_size, n_features))

cnn_time_distributed = TimeDistributed(
    base_model,
    input_shape=(timesteps, image_size, image_size, n_features)
)(inp)


#bidirectional convlstm2d appended to unet
lstm1 = Bidirectional(ConvLSTM2D(n_classes, kernel_size=(3, 3),
                    input_shape=(7, 128, 128, 4),
                    padding='same', return_sequences=True, recurrent_dropout = 0.3), 
                      merge_mode='ave')(cnn_time_distributed)
#bn1 = BatchNormalization()(lstm1)
out = Conv3D(n_classes, kernel_size=(4, 4, 4),
                activation='softmax',
                padding='same', data_format='channels_last')(lstm1)

lstm_model = Model(inp, out)

loss = CategoricalCELoss(class_weights=class_weights)

adam = Adam(lr=0.001, clipnorm=1.)
lstm_model.compile(optimizer=adam,
                   loss=loss,
                   metrics=['accuracy'])

# initialize data generators
path_to_data = 'F:/.../data/processed/training_data/'


############
# TIME SERIES IMAGE GENERATOR
############
def time_series_image_gen_v3(
        path_to_training_data,
        n_classes,
        split="train",
        batch_size=2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=45
        ):

    #aoi and associated planetscope imagery dates
    aoi_list = ["aoi0", "aoi1", "aoi2", "aoi3", "aoi4", "aoi5"]

    #only 12 dates for a time period
    aoi_dates_list = [['May082020','May122020','May202020','June012020','June052020','June072020','June122020','June172020','June252020','July032020','July122020','July192020'],
                      ['May082020','May122020','May202020','June012020','June052020','June072020','June122020','June172020','June252020','July122020','July152020','July182020'],
                      ['May152020','May192020','June012020','June032020','June052020','June142020','June162020','June272020','July102020','July122020','July182020','July242020'],
                      ['May082020','May122020','May152020','May202020','June032020','June052020','June072020','June172020','June252020','July032020','July122020','July302020'],
                      ['May122020','May202020','June032020','June052020','June072020','June122020','June172020','June252020','July022020','July122020','July182020','July302020'],
                      ['May122020','May202020','June012020','June062020','June172020','June252020','July082020','July102020','July122020','July192020','July242020','July302020']]

    
    files=[] #files[0] is aoi0:may032020, files[1] is aoi0:may082020...
    for a in range(len(aoi_dates_list)):
        #print(f'aoi{a}')
        #print(aoi_dates_list[a])
        for d in range(len(aoi_dates_list[a])):
            #print(aoi_dates_list[a][d])
            ps_folder_path = path_to_training_data+f'{aoi_list[a]}/{aoi_dates_list[a][d]}_data/{aoi_dates_list[a][d]}_{split}_tiles_(128x128)'
            files.append([x for x in glob(ps_folder_path+'/*.tif')])

    while True:
        
        timeseries_len = 12
        
        #initialize empty data containers for X, Y, and one-hot Y
        X = np.empty((batch_size, timeseries_len, 128, 128, n_features))
        Y = np.empty((batch_size, timeseries_len, 128, 128))
        Y_one_hot = np.empty((batch_size, timeseries_len, 128, 128, 7))
        #Y_one_hot_nobg = np.empty((batch_size, timeseries_len, 128, 128, 6))
        
        #loop through number of batches
        for b in range(batch_size):
            #print(f"batch  {b}")
            #randomly select AOI
            aoi_random_choice = np.random.randint(0, len(aoi_list), 1)
            selected_aoi = aoi_list[aoi_random_choice[0]]
            #print(selected_aoi)
            #subset files based on selected AOI
            #subset_files[0] is date 0 for selected aoi, subset_files[1] is date 1 for selected aoi, etc
            
            
            
            #12 dates subset
            if selected_aoi == "aoi0":
                subset_files = files[0:12]
            if selected_aoi == "aoi1":
                subset_files = files[12:24]
            if selected_aoi == "aoi2":
                subset_files = files[24:36]
            if selected_aoi == "aoi3":
                subset_files = files[36:48]
            if selected_aoi == "aoi4":
                subset_files = files[48:60]
            if selected_aoi == "aoi5":
                subset_files = files[60:72]
            
            #amount of timestep padding (zeros) to add before LSTM input
            lstm_padding_amount = timeseries_len - len(subset_files)      
            #print(f"padding {lstm_padding_amount}")
                
            #from files, randomly select one INDEX to use in time series stack
            file_random_choice = np.random.randint(0, len(subset_files[0]), 1)
            #select time series samples based on filepaths of index file_random_choice
            stacked_file_date_aoi = []
            for k in range(len(aoi_dates_list[aoi_random_choice[0]])):
                #stacked_files_dates[k] is the path for the same image at all timesteps
                stacked_file_date_aoi.append(subset_files[k][file_random_choice[0]])
    
            batch_X = []
            batch_Y = []
            
            for p in range(lstm_padding_amount):
                #print("adding padding")
                batch_X += [np.zeros(shape=((128, 128, n_features)))]
                batch_Y += [np.zeros(shape=((128, 128)))]
                
            for j, input_path in enumerate(stacked_file_date_aoi):
                #print("adding image")
                #print(input_path)
                image = np.array(imread(input_path, plugin="tifffile"), dtype=float)
                
                #image[:, :, :n_features] = preprocessing_image_ms_subset(image[:, :, :n_features])
                
                batch_X += [image[:, :, :4]]
                batch_Y += [image[:, :, 4]]
                        
            X[b, ...] = np.array(batch_X)
            Y[b, ...] = np.array(batch_Y)
            
            
            #Y_one_hot = tf.keras.utils.to_categorical(Y, num_classes=7)
            
            #Y_one_hot_nobg = Y_one_hot[:, :, :, :, 1:7]
            
        #yield(X, Y_one_hot)

            # # # image augmentation and one-hot-encoding
            # # # make sure to apply the same augmentation to every date
            # # # 1. outer loop: sample nr in batch
            # # # 2. random decision of image augmentation params
            # # # 3. inner loop: apply augmentation with params to all dates
            for s in range(batch_size):
                if horizontal_flip:
                    # randomly flip image up/down
                    if random.choice([True, False]):
                        for d in range(timeseries_len):
                            X = np.flipud(X)
                            Y = np.flipud(Y)
                            #for f in range(n_features):
                            #X[s, d, ..., f] = np.flipud(X[s, d, ..., f])
                                #Y[s, d, ...] = np.flipud(Y[s, d, ...])
                if vertical_flip:
                    # randomly flip image left/right
                    if random.choice([True, False]):
                        for d in range(timeseries_len):
                            X = np.fliplr(X)
                            Y = np.fliplr(Y)
                         
            Y_one_hot = tf.keras.utils.to_categorical(Y, num_classes=7)
        
        yield(X, Y_one_hot)


train_sequence_generator = time_series_image_gen_v3(
    path_to_data,
    n_classes=n_classes,
    split='train',
    batch_size=batch_size,
    horizontal_flip=True,
    vertical_flip=True
)

validation_sequence_generator = time_series_image_gen_v3(
    path_to_data,
    n_classes=n_classes,
    split='valid',
    batch_size=batch_size
)


tensorboard = TensorBoard(
    log_dir='F:\\...\\reports\\tensorboard' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    histogram_freq=1,
    embeddings_freq=1,
    profile_batch = 100000000
)


##Reduce Learning Rate on Plateau
def lr_schedule(epoch):
    learning_rate = 0.001
    if epoch > 10:
        learning_rate = 0.0005
    if epoch > 20:
        learning_rate = 0.00025
    if epoch > 30:
        learning_rate = 0.00001
    if epoch > 40:
        learning_rate = 0.0005
    if epoch > 50:
        learning_rate = 0.00001
        
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)


earlystopper = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=0.6,
    restore_best_weights=True
)

class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        
        plt.show();
        
plot_loss = PlotLearning()


#Epoch Checkpoints
filepath = r'F:/.../models/mymodel-{epoch:02d}-{accuracy:.3f}.h5'
mc = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=2)

start_time = time.time()

# train model
lstm_model.fit(
    train_sequence_generator,
    steps_per_epoch=200,
    epochs=100,
    verbose=1,
    validation_data=validation_sequence_generator,
    validation_steps=40, callbacks = [plot_loss])

end_time = time.time()






