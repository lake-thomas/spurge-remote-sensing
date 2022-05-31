

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

import segmentation_models as sm
SM_FRAMEWORK = tf.keras
sm.set_framework('tf.keras')
from segmentation_models import Unet
from segmentation_models.losses import CategoricalCELoss

# allow GPU memory growth
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

# set variables
dates = ['2020']

n_features = 8
image_size = 512
n_classes = 7
batch_size = 4

pixel_per_class = [1276564613, 138355011, 429326478, 153989042, 246958821, 5880996] #class weights based inverse proportional to number pixels per class

class_weights = np.array([0] + [1/nr for nr in pixel_per_class])
class_weights = class_weights/np.sum(class_weights)*1000

for i in range(len(class_weights)): 
    print(("%.17f" % class_weights[i]).rstrip('0').rstrip('.'))


# define model
model = Unet(
    backbone_name='seresnet50',
    encoder_weights=None,
    input_shape=(512, 512, n_features),
    classes=n_classes,
    activation='softmax'
)

loss = CategoricalCELoss(class_weights=class_weights)

adam = Adam(lr=0.0001)

model.compile(
    optimizer=adam,
    loss=loss,
    metrics=['accuracy']
)

def simple_image_generator(
        path_to_data,
        dates,
        split,
        n_classes,
        batch_size=4,
        rotation_range=0,
        horizontal_flip=False,
        vertical_flip=False
):
    """
    Adapted image generator from
    https://github.com/jensleitloff/CNN-Sentinel/blob/master/py/image_functions.py.
    Instead of the whole image, now each pixel is labelled with a one-hot-vector.
    """

    files = []
    for date in dates:
        files += glob(path_to_data+f'/{date}_data/{date}_{split}_tiles_(512x512)/*.tif')

    while True:
        # select batch_size number of samples without replacement
        batch_files = np.random.choice(files, batch_size, replace=True)

        # array for images
        batch_X = []
        batch_Y = []
        # loop over images of the current batch
        for _, input_path in enumerate(batch_files):
            image = np.array(io.imread(input_path), dtype=float)
            # scale sentinel bands
            #image[:, :, :n_features] = preprocessing_image_ms(image[:, :, :n_features])
            # process image
            if horizontal_flip:
                # randomly flip image up/down
                if random.choice([True, False]):
                    image = np.flipud(image)
            if vertical_flip:
                # randomly flip image left/right
                if random.choice([True, False]):
                    image = np.fliplr(image)
            # rotate image by random angle between
            # -rotation_range <= angle < rotation_range
            if rotation_range is not 0:
                angle = np.random.uniform(low=-abs(rotation_range),
                                          high=abs(rotation_range))
                image = rotate(image, angle, mode='reflect',
                               order=1, preserve_range=True)        
            # one hot encoding of labels
            # classes range from 1 to 5, but to_categorical counts from 0
            # therefore 0th index of last axis is omitted
            Y_one_hot = to_categorical(image[:, :, 8], num_classes=7)
            # put all together
            batch_X += [image[:, :, :8]]
            batch_Y += [Y_one_hot]
        # convert lists to np.array
        X = np.array(batch_X)
        Y = np.array(batch_Y)

        yield(X, Y)



# initialize data generators
path_to_data = 'D:/.../data/processed/training_data/'

train_sequence_generator = simple_image_generator(
    path_to_data,
    dates,
    n_classes=n_classes,
    split='train',
    batch_size=batch_size,
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True
)

validation_sequence_generator = simple_image_generator(
    path_to_data,
    dates,
    n_classes=n_classes,
    split='valid',
    batch_size=batch_size,
)


# define callbacks

#Reduce Learning Rate on Plateau
def lr_schedule(epoch):
    learning_rate = 0.001
    if epoch > 25:
        learning_rate = 0.0005
    if epoch > 50:
        learning_rate = 0.00025
    if epoch > 75:
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
filepath = r'D:/.../reports/mymodel-{epoch:02d}-{accuracy:.2f}.h5'
mc = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=10)

#Plot Loss and Accuracy
class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.f1 = []
        self.val_f1 = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.f1.append(logs.get('accuracy'))
        self.val_f1.append(logs.get('val_accuracy'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val loss")
        ax1.legend()
        
        ax2.plot(self.x, self.f1, label="Acc")
        ax2.plot(self.x, self.val_f1, label="val Acc ")
        ax2.legend()
        
        plt.show();
plot_losses = PlotLearning()



start_time = time.time()

#steps per epoch = training samples / batch size

model.fit(train_sequence_generator, steps_per_epoch=200,
                    validation_data=validation_sequence_generator, 
                    epochs=30, validation_steps=10,
                    callbacks = [plot_losses]) 

end_time = time.time()


