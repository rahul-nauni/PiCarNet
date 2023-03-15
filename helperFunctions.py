# -*- coding: utf-8 -*-
# Importing dependencies
import pandas as pd
import os
import numpy as np
from datetime import datetime
from glob import glob
from typing import List, Union, Iterator, Tuple
# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from keras.models import Sequential, Model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, History


def load_data(data_path: str) -> pd.DataFrame:
    """Loads data from csv file
    Params
    ------
    data_path: str
        Data path containing images and csv data
    Returns
    -------
    pd.DataFrame
        Loaded data as pandas dataframe
    """
    columns = ['image_id', 'angle', 'speed']
    data = pd.read_csv(os.path.join(data_path, 'training_norm.csv'), delimiter= ',', header=0, names = columns)

    return data

# Appends absolute path with image id column
def append_path(data: pd.DataFrame, data_path: os.PathLike) -> pd.DataFrame:
  data['image_id'] = data.image_id.apply(lambda x: os.path.join(data_path, 'training_data/', str(x)) + '.png')
  return data

"""## Data preprocessing"""

# train test split of dataset
def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Params
    ------
    data: pd.DataFrame
        Pandas dataframe containing all data.
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple of train and val split
    """
    rnd = np.random.RandomState(seed=None)

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=rnd.seed(1234)) # split data with test size of 20%

    return train_data, val_data

# plotting distribution of output labels
def visualize(data, attr):
    nBins = 60
    plt.hist(data.loc[:,attr], bins=nBins)
    plt.grid(False)
    plt.gca().set(title=f'Distribution of steering {attr}', ylabel='Frequency')

"""## Data pipeline"""

# Generate images on the fly while training model
def img_generator(train_data: pd.DataFrame, val_data: pd.DataFrame, test_dir: os.PathLike, BATCH_SIZE: int, IMG_HEIGHT: int, IMG_WIDTH: int) -> Tuple[Iterator, Iterator, Iterator]:
    """
    Params
    ------
    train_data: pd.DataFrame
        Pandas dataframe containing training data
    val_data: pd.DataFrame
        Pandas dataframe containing validation data
    test_dir: os.PathLike
        path to test data directory
    BATCH_SIZE: int
        Number of images to process in each batch.
    IMG_HEIGHT: int
        image height
    IMG_WIDTH: int
        image width
    Returns
    -------
    Tuple[Iterator, Iterator, Iterator]
        keras ImageDataGenerators used for training, validation and testing the model.
    """

    train_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.75, 1),
        shear_range=0.1,
        zoom_range=[0.75, 1],
        horizontal_flip=False,
        vertical_flip=False,
        validation_split=0.2,

    )

    val_generator = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_generator.flow_from_dataframe(dataframe=train_data,
                                                          directory=None,
                                                          x_col='image_id',
                                                          y_col=['angle','speed'],
                                                          color_mode='rgb',
                                                          target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                          class_mode='raw',
                                                          batch_size=BATCH_SIZE,
                                                          shuffle=True)

    val_generator = val_generator.flow_from_dataframe(dataframe=val_data,
                                                      x_col='image_id',
                                                      y_col=['angle', 'speed'],
                                                      color_mode='rgb',
                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                      class_mode='raw',
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=True)

    test_generator = test_generator.flow_from_directory(
        directory=test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='rgb',
        class_mode=None,
        classes=None,
        batch_size=BATCH_SIZE,
        seed=1234,
        shuffle=False)



    return train_generator, val_generator, test_generator

# create base model using transfer learning
def create_baseline_model(input_shape, dropout_rate, optimizer) -> Sequential:
    """Creates a baseline model from MobileNetV2

    Params
    ------
    input_shape: input dimensions of image (224x224x3).
    dropout_rate: Applies Dropout to the input, to prevent over-fitting.
    optimizer: Optimization algorithm.

    Returns
    -------
    Sequential
        The keras model.
    """
    mobilenet = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    mobilenet.trainable = False # Freeze the model

    # Rebuild top
    """
    model = Sequential([
        mobilenet,
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(56, activation='relu', use_bias=True),
        Dense(28, activation='relu', use_bias=True),
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(dropout_rate/2),
        Dense(8,activation='softmax', use_bias=False),
        Dense(2, activation='linear', use_bias=True)
    ], name='baseline')"""

    model = Sequential([
        mobilenet,
        Dropout(dropout_rate),
        Dense(64, activation='relu', use_bias=True),
        Dense(32, activation='relu', use_bias=True),
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(16, activation='relu', use_bias=True),
        Dense(2, activation='linear', use_bias=False)
    ], name='baseline')

    model.compile(loss=['mean_squared_error'], optimizer=optimizer, metrics=['mse', 'accuracy'])
    return model

# NOTE: we might need more complex model suggested by NVIDIA
# ref: https://towardsdatascience.com/teaching-cars-to-drive-using-deep-learning-steering-angle-prediction-5773154608f2

# defining callbacks
def get_callbacks(model: str) -> List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]:
    """Accepts the model name as a string and returns multiple keras callbacks

    Params
    ------
    model: str
        The name of model as a string

    Returns
    -------
    A list of multiple keras callbacks
    """
    logdir = (
        'logs/scalars/' + model + '_' + datetime.now().strftime('%Y%m%d-%H%M%S')
    ) # logging for each model
    tensorboard_callback = TensorBoard(log_dir=logdir)

    early_stopping_callback = EarlyStopping(
        monitor='loss',
        min_delta=0.1,  # model should improve by at least 0.1
        patience=10,  # amount of epochs  with improvements worse than 1% until the model stops
        verbose=1,
        mode='min',
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )

    model_checkpoint_callback = ModelCheckpoint(
        './data/models/' + model + '.h5',
        monitor='loss',
        verbose=0,
        save_best_only=True,  # save the best model
        mode='min',
        save_freq='epoch',  # save the model on disk at end of every epoch
    )
    return [model_checkpoint_callback, tensorboard_callback, early_stopping_callback]

## Training model
def train(name: str, model: Model, train_generator: Iterator, val_generator: Iterator, epoch: int, BATCH_SIZE: int) -> History:
    """
    Params
    ------
    name: str
        Model name
    model: Model
        Model we are training
    train_generator: Iterator
        Image data generator for training
    val_generator: Iterator
        Image data generator for validation
    epoch: int
        Number of epochs to train for
    BATCH_SIZE: int
        Number of images to process in each bacth
    Returns
    -------
    keras.callbacks.History
    """
    callbacks = get_callbacks(name)

    history = model.fit(train_generator,
              epochs=epoch,
              steps_per_epoch=300,
              validation_data=val_generator,
              validation_steps=100,
              callbacks=callbacks,
              workers=4,
              verbose=1,
              batch_size=BATCH_SIZE)

    return history

# Plotting training and validation loss from History callback
def plot_history(history):
    """Plots training and validation loss over epochs
    Params
    ------
    history: History
        keras History callback
    Returns
    -------
    None
    """
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    # As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    # Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'lightblue',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.3f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'coral',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.3f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(False)
    plt.legend()
    plt.savefig('tuned_baseline_plot.pdf')
    plt.show()

def get_predictions(test_path: os.PathLike, test_generator: Iterator, model: Sequential) -> pd.DataFrame:
    """
    Params
    ------
    test: os.PathLike
        path to test data
    test_generator: Iterator
        Data generator for testing
    model: Sequential model
        trained model to make

    Returns
    -------
    pd.DataFrame
        predictions as pandas dataframe
    """

    # list of all png files in test data
    png_files = glob(f'{test_path}/*png')

    # convert to pandas dataframe with image_id as column name
    png_df = pd.DataFrame(data=png_files, columns=['image_id'])
    png_df['image_id'] = png_df.image_id.apply(lambda x: os.path.split(x)[-1].split('.png')[0])

    """# prepare test tensorflow dataset for making predictions
    test_ds = tf.data.Dataset.from_tensor_slices(png_files).map(
        lambda image: (tf.image.decode_png(tf.io.read_file(image), channels=3))
    ).map(
        lambda image: (tf.image.convert_image_dtype(image, dtype=tf.float32))
    ).map(
        lambda image: (tf.image.resize(image, [224, 224]))
    ).batch(32)"""

    # make predictions
    pred = model.predict(test_generator)

    # convert prediction numpy.ndarray( to pd.DataFrame
    pred_df = pd.concat([png_df, pd.DataFrame(data=pred, columns=['angle', 'speed'])], axis=1)
    # dropping index from dataframe
    pred_df.reset_index(drop=True)

    # return predictions
    return pred_df

def save_csv(pred_df: pd.DataFrame) -> None:
    """
    Params
    ------
    pred_df: pd.DataFrame
        predictions in pandas dataframe

    Returns
    -------
    None
    """

    savedir='./submissions' # submissions directory
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    csvfile = 'submission' + '_' + datetime.now().strftime('%d-%b_%I-%M%p') + '.csv' # csv file name
    savedir = str(os.path.join(savedir, csvfile))
    pred_df.to_csv(savedir, sep=',', index=False) # save to disk
    print('Saved CSV file on disk!')