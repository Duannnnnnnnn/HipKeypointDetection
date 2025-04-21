# train7.py
# train_model.py with argparse (Refactored with Parameter Injection)
import os
import cv2
import math
import ctypes
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import json
from PIL import Image
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetV2S, EfficientNetV2M, EfficientNetB0, EfficientNetB1, EfficientNetB2
from sklearn.model_selection import train_test_split

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--log_base_dir", type=str, default="logs")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--early_patience", type=int, default=15)
    parser.add_argument("--model_name", type=str, default="efficientnetv2s", 
                        choices=["efficientnetv2s", "efficientnetv2m", "efficientnetb0", "efficientnetb1", "efficientnetb2"])
    parser.add_argument("--save_best", action="store_true", help="Save the best model based on val_loss")
    parser.add_argument("--use_lr_scheduler", action="store_true", help="Use ReduceLROnPlateau")
    return parser.parse_args()

def build_model(model_name, resolution):
    input_shape = (resolution, resolution, 3)
    if model_name == "efficientnetv2s":
        base = EfficientNetV2S(include_top=False, weights="imagenet", input_shape=input_shape)
    elif model_name == "efficientnetv2m":
        base = EfficientNetV2M(include_top=False, weights="imagenet", input_shape=input_shape)
    elif model_name == "efficientnetb0":
        base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
    elif model_name == "efficientnetb1":
        base = EfficientNetB1(include_top=False, weights="imagenet", input_shape=input_shape)
    elif model_name == "efficientnetb2":
        base = EfficientNetB2(include_top=False, weights="imagenet", input_shape=input_shape)
    else:
        raise ValueError("Unsupported model name")

    base.trainable = True
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(2048, activation='relu')(x)
    out = layers.Dense(24, activation='linear')(x)
    return Model(inputs=base.input, outputs=out)

huber_loss = tf.keras.losses.Huber(delta=1.0)

def custom_mae(y_true, y_pred):
    return tf.keras.losses.MeanAbsoluteError()(y_true[:, :24], y_pred)

def custom_mse(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true[:, :24], y_pred)

@tf.function
def combined_first_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    img_sizes = y_true[:, -2:]
    y_true_pts = tf.reshape(y_true[:, :24], (-1, 12, 2))
    y_pred_pts = tf.reshape(y_pred, (-1, 12, 2))
    abs_loss = huber_loss(y_true_pts, y_pred_pts)
    img_sizes_r = tf.reshape(img_sizes, (-1, 1, 2))
    rel_loss = tf.reduce_mean(tf.abs(y_true_pts / img_sizes_r - y_pred_pts / img_sizes_r))
    center_loss = tf.reduce_mean(tf.norm(tf.reduce_mean(y_true_pts[:, :6], axis=1) - tf.reduce_mean(y_pred_pts[:, :6], axis=1), axis=1))
    return abs_loss + rel_loss + 0.2 * center_loss

if __name__ == "__main__":
    args = parse_arguments()
    from train5 import main

    model_filename = f"model_e{args.epochs}_bs{args.batch_size}_lr{args.lr}_p{args.early_patience}_{args.model_name}.keras"
    model_path = os.path.join(args.model_dir, model_filename)

    callbacks = [EarlyStopping(monitor='val_loss', patience=args.early_patience, restore_best_weights=True)]
    if args.save_best:
        callbacks.append(ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True))
    if args.use_lr_scheduler:
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, min_lr=1e-5))

    main(args, callbacks=callbacks)
