#Training Baseline CNN Model with AOI2 2020 Imagery
        
       
"""
CNN Model with Single-Date Satellite Imagery
Test parameters:
- backbone: SE-ResNet50
- kernel_size:  3
- dates: 2020
"""


# packages
import os
import sys
import time
import random
from glob import glob
import numpy as np
from skimage import io
from skimage.io import imsave, imread, imshow
from skimage.transform import rotate
from sklearn.metrics import multilabel_confusion_matrix, cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score, fbeta_score
import matplotlib.pyplot as plt
from functools import partial, update_wrapper
from itertools import product
from IPython.display import HTML, display, clear_output
from tabulate import tabulate
import time
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.python.keras.layers import *

import segmentation_models as sm
SM_FRAMEWORK = tf.keras
sm.set_framework('tf.keras')
from segmentation_models import Unet
from segmentation_models.losses import CategoricalCELoss

sys.path.append('C:/Users/.../PythonScripts/')

# allow GPU memory growth
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)


n_features = 4
image_size = 128
n_classes = 7
batch_size = 8


class_weights = [0, 0.0066, 0.0562, 0.0182, 0.0447, 0.0569, 0.8142] #proportional to pixel frequency and sums to ~1 (sum to ~1 so no effect on learning rate)

for i in range(len(class_weights)): 
    print(("%.17f" % class_weights[i]).rstrip('0').rstrip('.'))

# define model
model = Unet(
    backbone_name='seresnet50',
    encoder_weights=None,
    input_shape=(128, 128, n_features),
    classes=n_classes,
    activation='softmax'
)

loss = CategoricalCELoss(class_weights=class_weights)
#loss = DiceLoss(beta=1, class_weights=class_weights)
adam = Adam(lr=0.0001)

model.compile(
    optimizer=adam,
    loss=loss,
    metrics=['accuracy']
)


############
# IMAGE GENERATOR
############
def image_gen_v3(
        path_to_training_data,
        split="train",
        batch_size=2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=45
        ):
    
    #function testing parameters
    #path_to_training_data = 'F:/.../data/processed/training_data/'
    #split='train'
    #horizontal_flip=True
    #vertical_flip=True
    #rotation_range=45

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

        #initialize empty data containers for X, Y, and one-hot Y
        X = np.empty((batch_size, 128, 128, n_features))
        Y = np.empty((batch_size, 128, 128))
        Y_one_hot = np.empty((batch_size, 128, 128, 7))
        #Y_one_hot_nogb = np.empty((batch_size, 128, 128, 6))
        
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
           
                
            #from files, randomly select one INDEX to use in time series stack
            file_random_choice = np.random.randint(0, len(subset_files[0]), 1)
            date_random_choice = np.random.randint(0, len(subset_files), 1)
            #select time series samples based on filepaths of index file_random_choice
            input_path = subset_files[date_random_choice[0]][file_random_choice[0]]
            
            batch_X = []
            batch_Y = []
            
            image = np.array(imread(input_path, plugin="tifffile"), dtype=float)
            
            
            if horizontal_flip:
                if random.choice([True, False]):
                    image = np.flipud(image)
            if vertical_flip:
                if random.choice([True, False]):
                    image = np.fliplr(image)
            if rotation_range is not 0:
                angle = np.random.uniform(low=-abs(rotation_range),high=abs(rotation_range))
                image = rotate(image, angle, mode='reflect', order=1, preserve_range=True)        
            batch_X += [image[:, :, :4]]
            batch_Y += [image[:, :, 4]]
                        
            X[b, ...] = np.array(batch_X)
            Y[b, ...] = np.array(batch_Y)
            
            Y_one_hot = tf.keras.utils.to_categorical(Y, num_classes=7)
            #remove background class
            #Y_one_hot_nobg = Y_one_hot[:, :, :, 1:7]

        yield(X, Y_one_hot)

# initialize data generators
path_to_data = 'F:/.../data/processed/training_data/'

train_sequence_generator = image_gen_v3(
    path_to_data,
    split='train',
    batch_size=batch_size,
    rotation_range=0,
    horizontal_flip=True,
    vertical_flip=True
)

validation_sequence_generator = image_gen_v3(
    path_to_data,
    split='valid',
    batch_size=batch_size,
)


# define callbacks


#Reduce Learning Rate on Plateau
def lr_schedule(epoch):
    learning_rate = 0.001
    if epoch > 50:
        learning_rate = 0.0005
    if epoch > 70:
        learning_rate = 0.00025
    if epoch > 80:
        learning_rate = 0.00001
    if epoch > 100:
        learning_rate = 0.0005
    if epoch > 150:
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
    restore_best_weights=False
)

tensorboard = TensorBoard(
    log_dir='D:\\...\\reports\\tensorboard' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    write_graph=True,
    write_grads=True,
    write_images=True,
    update_freq='epoch',
    histogram_freq=1,
    profile_batch = 100000000
)

#Epoch Checkpoints
filepath = r'F:/.../models/mymodel-{epoch:02d}-{accuracy:.2f}.h5'
mc = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=10)

#Plot Loss and Accuracy
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



start_time = time.time()

model.fit(train_sequence_generator, steps_per_epoch=200,
                    validation_data=validation_sequence_generator, 
                    epochs=10, validation_steps=40,
                    callbacks = [plot_loss])

end_time = time.time()

model.save(r'F:/my_trained_model.h5')
