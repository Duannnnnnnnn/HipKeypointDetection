# train_model.py with argparse
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
from tensorflow.keras.applications import EfficientNetV2S, EfficientNetV2M, EfficientNetB0, EfficientNetB1, EfficientNetB2
from sklearn.model_selection import train_test_split

# === argparse 參數解析 ===
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--resolution", type=int, default=224)
parser.add_argument("--log_base_dir", type=str, default="logs")
parser.add_argument("--model_dir", type=str, default="models")
parser.add_argument("--early_patience", type=int, default=15)
parser.add_argument("--model_name", type=str, default="efficientnetv2s", 
                    choices=["efficientnetv2s", "efficientnetv2m", "efficientnetb0", "efficientnetb1", "efficientnetb2"])
args = parser.parse_args()

# === 超參數區 ===
resolution = args.resolution
batch_size = args.batch_size
initial_lr = args.lr
train_epochs = args.epochs
patience = args.early_patience
val_split = 0.15
test_split = 0.1
shuffle_seed = 42
log_base_dir = args.log_base_dir
model_dir = args.model_dir
experiment_name = f"e{train_epochs}_bs{batch_size}_lr{initial_lr}_p{patience}_{args.model_name}"
log_dir = os.path.join(log_base_dir, experiment_name)
model_filename = f"model_{experiment_name}.keras"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# 載入 DLL（修復 OpenCV 問題）
dll_path = os.path.abspath("tf_gpu_env/Lib/site-packages/cv2/zlibwapi.dll")
ctypes.WinDLL(dll_path)
print("DLL loaded from:", dll_path)

# 資料路徑與設定
images_path = "./xray_IHDI_1/images/"
labels_path = "./xray_IHDI_1/annotations/"

# 資料集切分
image_filenames = os.listdir(images_path)
train_val_filenames, test_filenames = train_test_split(image_filenames, test_size=test_split, random_state=shuffle_seed)
train_filenames, val_filenames = train_test_split(train_val_filenames, test_size=val_split, random_state=shuffle_seed)

# 模型構建函式
def build_model(model_name):
    if model_name == "efficientnetv2s":
        base = EfficientNetV2S(include_top=False, weights="imagenet", input_shape=(resolution, resolution, 3))
    elif model_name == "efficientnetv2m":
        base = EfficientNetV2M(include_top=False, weights="imagenet", input_shape=(resolution, resolution, 3))
    elif model_name == "efficientnetb0":
        base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(resolution, resolution, 3))
    elif model_name == "efficientnetb1":
        base = EfficientNetB1(include_top=False, weights="imagenet", input_shape=(resolution, resolution, 3))
    elif model_name == "efficientnetb2":
        base = EfficientNetB2(include_top=False, weights="imagenet", input_shape=(resolution, resolution, 3))
    else:
        raise ValueError("Unsupported model name")

    base.trainable = True
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(2048, activation='relu')(x)
    out = layers.Dense(24, activation='linear')(x)
    return Model(inputs=base.input, outputs=out)

model = build_model(args.model_name)

# 儲存實驗統計資訊
summary = {
    "experiment_name": experiment_name,
    "model_name": args.model_name,
    "batch_size": batch_size,
    "initial_lr": initial_lr,
    "epochs_requested": train_epochs,
    "early_patience": patience,
    "resolution": resolution
}

# === Pixel Error 計算函式 ===
def compute_pixel_error(model, test_filenames):
    errors = []
    for name in test_filenames:
        img = Image.open(os.path.join(images_path, name)).convert("L")
        orig_size = img.size
        img = img.resize((resolution, resolution))
        arr = np.array(img, dtype=np.float32) / 255.
        arr = np.stack([arr]*3, axis=-1)
        inp = np.expand_dims(arr, axis=0)

        pred = model.predict(inp)[0].reshape(-1, 2)
        pred[:, 0] *= orig_size[0] / resolution
        pred[:, 1] *= orig_size[1] / resolution

        df = pd.read_csv(os.path.join(labels_path, name.replace(".jpg", ".csv")), header=None)
        true = []
        for v in df.values.flatten():
            x, y = map(float, v.strip("()").split(","))
            true.append((x, y))
        true = np.array(true)

        dist = np.linalg.norm(pred - true, axis=1)
        errors.append(dist.mean())
    return np.mean(errors)


# 資料處理

def load_labels(img):
    df = pd.read_csv(os.path.join(labels_path, img.replace('.jpg', '.csv')), header=None)
    labels = []
    for val in df.values.flatten():
        x, y = map(float, val.strip('()').split(','))
        labels.extend([x, y])
    return np.array(labels)

def load_image(img, size=(resolution, resolution)):
    path = os.path.join(images_path, img)
    im = Image.open(path).convert('L')
    orig_size = im.size
    im = im.resize(size)
    arr = np.array(im, dtype=np.float32) / 255.
    return np.stack([arr]*3, axis=-1), orig_size

def preprocess_labels(labels, orig_size, target_size=(resolution, resolution)):
    x_r = target_size[0] / orig_size[0]
    y_r = target_size[1] / orig_size[1]
    return labels * np.array([x_r, y_r] * (len(labels) // 2))

# 資料增強 + Generator
def data_generator(files, batch_size=batch_size, shuffle=False):
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.03, height_shift_range=0.03)
    while True:
        if shuffle: random.shuffle(files)
        imgs, labels = [], []
        for f in files:
            img, orig = load_image(f)
            lab = preprocess_labels(load_labels(f), orig)
            tfm = datagen.get_random_transform((resolution, resolution))
            img = datagen.apply_transform(img, tfm)
            lab = lab.reshape(1, 24).astype(np.float32)
            orig = np.array(orig).reshape(1, 2).astype(np.float32)
            labels.append(np.concatenate([lab, orig], axis=1))
            imgs.append(img)
            if len(imgs) == batch_size:
                yield np.array(imgs), np.array(labels).reshape(-1, 26)
                imgs, labels = [], []
        if imgs:
            yield np.array(imgs), np.array(labels).reshape(-1, 26)

# Loss 與 metric
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

# 訓練模型
model.compile(optimizer=tf.keras.optimizers.Adam(initial_lr),
              loss=combined_first_loss,
              metrics=[custom_mae, custom_mse])

steps_per_epoch = math.ceil(len(train_filenames) / batch_size)
val_steps = math.ceil(len(val_filenames) / batch_size)

# tf.data.Dataset 包裝

def create_tf_dataset(files, batch_size=batch_size, shuffle=False):
    return tf.data.Dataset.from_generator(
        lambda: data_generator(files, batch_size, shuffle=shuffle),
        output_signature=(
            tf.TensorSpec(shape=(None, resolution, resolution, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 26), dtype=tf.float32)
        ))

train_ds = create_tf_dataset(train_filenames, shuffle=True)
val_ds = create_tf_dataset(val_filenames)

# 訓練
history = model.fit(train_ds,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=val_ds,
                    validation_steps=val_steps,
                    epochs=train_epochs,
                    callbacks=[
                        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
                    ])

# 儲存模型
model_path = os.path.join(model_dir, f"model_{experiment_name}.keras")
model.save(model_path)
avg_pixel_error = compute_pixel_error(model, test_filenames)
summary["avg_pixel_error"] = avg_pixel_error
print(f"✅ Average pixel error on test set: {avg_pixel_error:.2f} px")

# 訓練趨勢可視化並保存，並導出 csv log

def save_training_plot(history, filename_prefix):
    pd.DataFrame(history.history).to_csv(os.path.join(log_dir, f"{filename_prefix}_history.csv"), index=False)

    # === Log scale Loss 圖 ===
    plt.figure()
    plt.plot(np.log(np.array(history.history['loss']) + 1e-6), label='Training Loss (log)')
    if 'val_loss' in history.history:
        plt.plot(np.log(np.array(history.history['val_loss']) + 1e-6), label='Validation Loss (log)')
    plt.xlabel('Epochs')
    plt.ylabel('log(Loss)')
    plt.legend()
    plt.savefig(os.path.join(log_dir, f"{filename_prefix}_loss_log.png"))
    plt.close()

    # === Log MAE / MSE 圖 ===
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.log(np.array(history.history['custom_mae']) + 1e-6), label='Train MAE (log)')
    plt.plot(np.log(np.array(history.history['val_custom_mae']) + 1e-6), label='Val MAE (log)')
    plt.title('Log MAE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.log(np.array(history.history['custom_mse']) + 1e-6), label='Train MSE (log)')
    plt.plot(np.log(np.array(history.history['val_custom_mse']) + 1e-6), label='Val MSE (log)')
    plt.title('Log MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"{filename_prefix}_mae_mse_log.png"))
    plt.close()

    # 額外提示 EarlyStopping 若中途停止
    if len(history.history['loss']) < train_epochs:
        print(f"🛑 Training stopped early at epoch {len(history.history['loss'])}/{train_epochs} due to EarlyStopping.")

save_training_plot(history, f"{experiment_name}")
summary["epochs_trained"] = len(history.history["loss"])
summary["final_val_loss"] = history.history["val_loss"][-1]
summary["final_val_mae"] = history.history["val_custom_mae"][-1]
summary["final_val_mse"] = history.history["val_custom_mse"][-1]
summary["best_val_loss"] = np.min(history.history["val_loss"])
with open(os.path.join(log_dir, "summary.json"), "w") as f: json.dump(summary, f, indent=2)
