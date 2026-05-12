import argparse
import logging
import numpy as np
import os
import socket
import pdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

hostname = socket.gethostname()
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/quahb/.conda/pkgs/cuda-nvcc-12.1.105-0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' # dont use the entire gpu(s)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import yaml

from tqdm import tqdm

from models import get_model, get_training_cb
from utils import np_to_tfdataset, load_dataset

CONFIG='train_all'
BATCH_SIZE = 16
TRAIN_SIZE = 0.8
LEARNING_RATE = 0.001
PATIENCE = 5
EPOCHS = 40

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)
    print(f'Saving training outputs to {args.output_folder}')
    
    # start: data loading
    print(f'Loading training dataset from: {args.input_folder}')
    images_path = os.path.join(args.input_folder, 'images')
    labels_path = os.path.join(args.input_folder, 'labels')

    images = load_dataset(os.path.join(args.input_folder, 'images'))
    labels = load_dataset(os.path.join(args.input_folder, 'labels'))
    
    print(f'Images, Labels dimensions: {images.shape}, {labels.shape}')

    shuffle = np.random.RandomState(seed=42).permutation(len(images))
    images, labels = images[shuffle], labels[shuffle]

    print(f"Splitting whole dataset into train/validation: {TRAIN_SIZE}/{1.0 - TRAIN_SIZE}")
    val_i = int( len(images) * TRAIN_SIZE )
    X_train, y_train = images[:val_i], labels[:val_i]
    X_valid, y_valid = images[val_i:], labels[val_i:]

    del images, labels

    print('Train/Valid split:')
    print('{}, {}, {}, {}'.format(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape))
    
    # start: network loading
    print('Available GPUs:')
    for i in tf.config.list_physical_devices('GPU'): print(f'    {i}')

    gpus = [ '/GPU:0', '/GPU:1', '/GPU:2', '/GPU:3' ]
    strategy = tf.distribute.MirroredStrategy(devices=gpus)
    with strategy.scope():
        train_data = np_to_tfdataset(X_train, y_train, batch_size=BATCH_SIZE)
        if TRAIN_SIZE < 1.0:
            val_data = np_to_tfdataset(X_valid, y_valid, batch_size=BATCH_SIZE)
        else:
            val_data = None
        print('Creating model...')
        
        # network parameters
        dimensions = 2
        input_shape = X_train.shape[1:]
        loss_fn = 'mse'
        
        model = get_model(
                dimensions,
                input_shape,
                loss_fn,
                LEARNING_RATE,          
        )

    model.summary(print_fn=logging.debug)

    cb_list = get_training_cb(
            CONFIG,
            patience=PATIENCE,
            save_path=args.output_folder
    )

    history = model.fit(
            train_data,
            validation_data=val_data,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            initial_epoch=0,
            callbacks=cb_list,
            shuffle=True
    )

    print(history.history)
    print(f'Epochs Trained: {len(history.history)}, Patience: {PATIENCE}')

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder')
    parser.add_argument('output_folder')
    
    return parser

if __name__ == '__main__':
    main()
