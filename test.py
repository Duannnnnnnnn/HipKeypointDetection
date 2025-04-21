#2025/3/19 修改datagen回之前的width_range=height_range=0.03，調整顯示平均差距圖片

# Step 1: Setup Google Colab environment and install necessary libraries
# Install TensorFlow and EfficientNet libraries
# Import the required packages
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import random
import math
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras import layers, Model
import os
import ctypes

# 指定要載入的 DLL 絕對路徑
dll_path = os.path.abspath("tf_gpu_env/Lib/site-packages/cv2/zlibwapi.dll")
ctypes.WinDLL(dll_path)
print("DLL loaded from:", dll_path)

# Step 2: Data Preprocessing
# Load dataset, which consists of images and the coordinates of 12 key points for the hip
resolution = 224

# 更改開始(我將正確的資料集上傳到_1資料集)
dataset_path = "./xray_IHDI_1/images/"
images_path = "./xray_IHDI_1/images/"
labels_folder_path = "./xray_IHDI_1/annotations/"
# 更改結束

# 建構模型
def build_model():
    base_model = EfficientNetV2S(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base_model.trainable = True

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(2048, activation='relu')(x)
    outputs = layers.Dense(24, activation='linear')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model  # <-- 回傳兩個


model, base_model = build_model()
# model.summary()


# Load CSV labels for each image
def load_labels(image_name):
    label_path = os.path.join(labels_folder_path, image_name.replace('.jpg', '.csv'))  # Assuming label filenames match image filenames
    labels_df = pd.read_csv(label_path, header=None)
    labels = labels_df.values.flatten()
    parsed_labels = []
    for label in labels:
        label = label.strip('()')
        x, y = map(float, label.split(','))
        parsed_labels.extend([x, y])
    return np.array(parsed_labels)



# Split the dataset into training and validation
image_filenames = os.listdir(images_path)
# 確認圖片張數
#num_images = len(image_filenames)
#print(f"總共有 {num_images} 張圖片")

# 先將資料集切分出測試集 (例如 10% 用於測試)
train_val_filenames, test_filenames = train_test_split(image_filenames, test_size=0.1, random_state=42)

# 再從剩下的資料中切分出驗證集 (例如 15% 用於驗證，85% 用於訓練)
train_filenames, val_filenames = train_test_split(train_val_filenames, test_size=0.15, random_state=42)

from PIL import Image

def load_image(image_name, target_size=(resolution, resolution)):
    image_path = os.path.join(images_path, image_name)
    
    # 讀入圖片（灰階），並轉成 RGB 三通道
    image = Image.open(image_path).convert('L')
    original_size = image.size  # (width, height)
    image = image.resize(target_size)
    
    # 轉成 numpy 格式並正規化 [0,1]
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.stack([image] * 3, axis=-1)  # (H, W, 3)
    
    return image, original_size



# Adjust labels to match image size (preprocess)
def preprocess_labels(labels, original_size, target_size=(resolution, resolution)):
    x_ratio = target_size[0] / original_size[0]
    y_ratio = target_size[1] / original_size[1]
    return labels.astype(float) * np.array([x_ratio, y_ratio] * (len(labels) // 2))

# Adjust labels to match image
def transform_labels(labels, transform_params, target_size):
    labels = labels.reshape(-1, 2).astype(np.float64)  # 確保高精度運算  # 假設一維 [x1, y1, x2, y2...]，轉換為 (N, 2)


    # 計算影像的中心 (旋轉基準點)
    center_x = target_size[0] / 2
    center_y = target_size[1] / 2


    # 旋轉
    angle = transform_params.get('theta', 0)
    if angle != 0:
        radians = np.deg2rad(-angle)
        cos_theta, sin_theta = np.cos(radians), np.sin(radians)

        # 先平移到中心點
        labels[:, 0] -= center_x
        labels[:, 1] -= center_y

        # 旋轉
        rotated_x = labels[:, 0] * cos_theta - labels[:, 1] * sin_theta
        rotated_y = labels[:, 0] * sin_theta + labels[:, 1] * cos_theta

        # 再平移回來
        labels[:, 0] = rotated_x + center_x
        labels[:, 1] = rotated_y + center_y


    # 平移
    labels[:, 0] -= transform_params.get('tx', 0)
    labels[:, 1] -= transform_params.get('ty', 0)


    # 確保標註點不超出範圍
    labels[:, 0] = np.clip(labels[:, 0], 0, target_size[1] - 1)
    labels[:, 1] = np.clip(labels[:, 1], 0, target_size[0] - 1)

    return labels.flatten()


# Generator function to yield batches of data
def data_generator(filenames, batch_size=8, target_size=(224,224), shuffle=False):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.03,
        height_shift_range=0.03,
        fill_mode='nearest'
    ) # Reduced batch size for stability
    while True:
        if shuffle:
            random.shuffle(filenames)
        batch_images = []
        batch_labels = []
        for filename in filenames:
            image_name = filename
            labels = load_labels(filename)

            # Load and preprocess image and labels
            image, original_size = load_image(filename, target_size=target_size)

            # 獲取隨機增強參數
            transform_params = datagen.get_random_transform(target_size)
            #print(f"[DEBUG] Transform Params: {transform_params}")  # 顯示轉換參數

            # 對影像進行隨機增強
            image = datagen.apply_transform(image, transform_params)
            labels = preprocess_labels(labels, original_size)

            # 強制 shape 為 (1,24)
            labels = np.array(labels, dtype=np.float32).reshape(1, 24)
            # 強制原始尺寸為 (1,2)
            original_size = np.array(original_size, dtype=np.float32).reshape(1, 2)
            # 合併成 (1,26)
            labels_with_size = np.concatenate([labels, original_size], axis=1)

            batch_images.append(image)
            batch_labels.append(labels_with_size)
            if len(batch_images) == batch_size:
                batch_images = np.array(batch_images, dtype=np.float32)
                batch_labels = np.array(batch_labels, dtype=np.float32).reshape(-1, 26)

                yield batch_images, batch_labels
                batch_images, batch_labels = [], []
        if len(batch_images) > 0:
            batch_images = np.array(batch_images, dtype=np.float32)
            batch_labels = np.array(batch_labels, dtype=np.float32).reshape(-1, 26)

            yield batch_images, batch_labels
            batch_images, batch_labels = [], []



# Step 3: Model Definition
# Load EfficientNetB0 model and modify the output for keypoint regression
# base_model = efn.EfficientNetB0(input_shape=(resolution, resolution, 3), include_top=False, weights='imagenet')
# base_model.trainable = True

# 自訂分類：先 Global Average Pooling，再一個 2048 維的全連接層（ReLU），再輸出 24 維
# global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
# dense_1 = tf.keras.layers.Dense(2048, activation='relu')(global_avg_pool)
# output_layer = tf.keras.layers.Dense(24, activation='linear')(dense_1)

# # 組合模型：輸入依然是 EfficientNet 的輸入，輸出為新的 24 維分類頭
# model = tf.keras.models.Model(inputs=base_model.input, outputs=output_layer)

# loss function
def huber_loss(y_true, y_pred):
    """ 計算單點位置的 Huber Loss """
    return tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)

def center_alignment_loss_group_of_three(y_true, y_pred, img_sizes):
    """
    計算 12 個點分成 4 組（每組 3 個點）後，
    預測和真實關鍵點群組中心之間的相對誤差。

    輸入:
      y_true: shape = (batch_size, 12, 2) 真實關鍵點座標
      y_pred: shape = (batch_size, 12, 2) 預測關鍵點座標
      img_sizes: shape = (batch_size, 2) 圖片原始尺寸 (width, height)

    計算方式:
      對每一組（三個點）計算預測與真實的群組中心，
      然後對每個群組計算：x 差距除以圖片寬度、y 差距除以圖片高度，
      最後使用 L2 節距計算該組的歸一化誤差，再取平均。
    """
    group_errors = []
    # 12 個點分成 4 組，每組 3 個點
    for i in range(0, 12, 3):
        # 取出第 i 到 i+3 個點
        true_center = tf.reduce_mean(y_true[:, i:i+3, :], axis=1)  # shape (batch_size, 2)
        pred_center = tf.reduce_mean(y_pred[:, i:i+3, :], axis=1)  # shape (batch_size, 2)
        delta = true_center - pred_center  # shape (batch_size, 2)
        # 歸一化誤差：x 分別除以圖片寬度，y 除以圖片高度
        normalized_error = tf.sqrt(tf.square(delta[:, 0] / img_sizes[:, 0]) +
                                   tf.square(delta[:, 1] / img_sizes[:, 1]))  # shape (batch_size,)
        group_errors.append(normalized_error)
    # 將每組的誤差堆疊起來，然後取平均
    group_errors = tf.stack(group_errors, axis=1)  # shape (batch_size, 4)
    return tf.reduce_mean(group_errors)


def center_alignment_loss(y_true, y_pred, img_sizes):
    """
    計算左右兩側（各 6 個點）的重心，並比較預測與真實重心的相對誤差，
    相對誤差的計算方式為：x 差距除以圖片寬度、y 差距除以圖片高度，再使用 L2 節距計算整體誤差。

    輸入:
      y_true: shape = (batch_size, 12, 2)  真實關鍵點座標
      y_pred: shape = (batch_size, 12, 2)  預測關鍵點座標
      img_sizes: shape = (batch_size, 2)  圖片原始尺寸 (width, height)
    """
    # 分別計算左右兩側的重心
    true_center_left = tf.reduce_mean(y_true[:, :6, :], axis=1)   # (batch_size, 2)
    true_center_right = tf.reduce_mean(y_true[:, 6:, :], axis=1)   # (batch_size, 2)
    pred_center_left = tf.reduce_mean(y_pred[:, :6, :], axis=1)     # (batch_size, 2)
    pred_center_right = tf.reduce_mean(y_pred[:, 6:, :], axis=1)     # (batch_size, 2)

    # 分別計算左右重心的差異
    delta_left = true_center_left - pred_center_left  # (batch_size, 2)
    delta_right = true_center_right - pred_center_right  # (batch_size, 2)

    # 分別將 x 差除以圖片寬，y 差除以圖片高
    norm_left = tf.sqrt(tf.square(delta_left[:, 0] / img_sizes[:, 0]) + tf.square(delta_left[:, 1] / img_sizes[:, 1]))
    norm_right = tf.sqrt(tf.square(delta_right[:, 0] / img_sizes[:, 0]) + tf.square(delta_right[:, 1] / img_sizes[:, 1]))

    # 平均每個樣本的左右誤差
    loss_left = tf.reduce_mean(norm_left)
    loss_right = tf.reduce_mean(norm_right)

    return loss_left + loss_right


@tf.function
def combined_first_loss(y_true, y_pred):
    """
    綜合 Loss：包含絕對位置誤差（Huber Loss）以及相對位置誤差（相對於圖片原始尺寸）。

    輸入:
      y_true: shape = (batch_size, 26) —— 前 24 維為關鍵點座標，後 2 維為圖片寬高
      y_pred: shape = (batch_size, 24) —— 預測 12 個點的 (x, y)

    算法:
      1. 從 y_true 中拆分出關鍵點和圖片尺寸。
      2. 將關鍵點重塑成 (batch_size, 12, 2)。
      3. 計算絕對位置誤差：使用 Huber Loss 比較預測點和真實點的 (x, y)。
      4. 計算相對位置誤差：將每個點的 x 座標除以圖片原始寬度，y 座標除以原始高度，
         得到歸一化的預測和真實座標，再計算它們的 L1 誤差。
      5. 返回兩部分的加權和（這裡權重可以根據需要調整）。
    """
    # 將輸入轉換為 float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # 拆分 y_true：前 24 維為關鍵點，後 2 維為圖片尺寸
    img_sizes = y_true[:, -2:]       # shape: (batch_size, 2)
    y_true_points = y_true[:, :24]     # shape: (batch_size, 24)

    # 重塑關鍵點數據為 (batch_size, 12, 2)
    y_true_points = tf.reshape(y_true_points, (-1, 12, 2))
    y_pred_points = tf.reshape(y_pred, (-1, 12, 2))

    # (1) 絕對位置誤差：使用 Huber Loss 計算
    abs_loss = tf.keras.losses.Huber(delta=1.0)(y_true_points, y_pred_points)

    # (2) 相對位置誤差：
    # 將每個點的 x 座標除以該圖片的寬，y 座標除以該圖片的高
    # 這裡利用 broadcasting，先把 img_sizes 重塑為 (batch_size, 1, 2)
    img_sizes_reshaped = tf.reshape(img_sizes, (-1, 1, 2))
    y_true_norm = y_true_points / img_sizes_reshaped
    y_pred_norm = y_pred_points / img_sizes_reshaped
    rel_loss = tf.reduce_mean(tf.abs(y_true_norm - y_pred_norm))

    # 重心對齊誤差：使用剛剛修改的版本，注意這裡需要傳入 img_sizes
    center_loss = center_alignment_loss(y_true_points, y_pred_points, img_sizes)


    # 返回總 Loss，這裡我們簡單地將兩部分相加（可根據需要調整權重）
    return abs_loss + rel_loss + 0.2 * center_loss


@tf.function
def combined_loss(y_true, y_pred):
    """
    綜合 Loss：包含絕對位置誤差（Huber Loss）以及相對位置誤差（相對於圖片原始尺寸）。

    輸入:
      y_true: shape = (batch_size, 26) —— 前 24 維為關鍵點座標，後 2 維為圖片寬高
      y_pred: shape = (batch_size, 24) —— 預測 12 個點的 (x, y)

    算法:
      1. 從 y_true 中拆分出關鍵點和圖片尺寸。
      2. 將關鍵點重塑成 (batch_size, 12, 2)。
      3. 計算絕對位置誤差：使用 Huber Loss 比較預測點和真實點的 (x, y)。
      4. 計算相對位置誤差：將每個點的 x 座標除以圖片原始寬度，y 座標除以原始高度，
         得到歸一化的預測和真實座標，再計算它們的 L1 誤差。
      5. 返回兩部分的加權和（這裡權重可以根據需要調整）。
    """
    # 將輸入轉換為 float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # 拆分 y_true：前 24 維為關鍵點，後 2 維為圖片尺寸
    img_sizes = y_true[:, -2:]       # shape: (batch_size, 2)
    y_true_points = y_true[:, :24]     # shape: (batch_size, 24)

    # 重塑關鍵點數據為 (batch_size, 12, 2)
    y_true_points = tf.reshape(y_true_points, (-1, 12, 2))
    y_pred_points = tf.reshape(y_pred, (-1, 12, 2))

    # (1) 絕對位置誤差：使用 Huber Loss 計算
    abs_loss = tf.keras.losses.Huber(delta=1.0)(y_true_points, y_pred_points)

    # (2) 相對位置誤差：
    # 將每個點的 x 座標除以該圖片的寬，y 座標除以該圖片的高
    # 這裡利用 broadcasting，先把 img_sizes 重塑為 (batch_size, 1, 2)
    img_sizes_reshaped = tf.reshape(img_sizes, (-1, 1, 2))
    y_true_norm = y_true_points / img_sizes_reshaped
    y_pred_norm = y_pred_points / img_sizes_reshaped
    rel_loss = tf.reduce_mean(tf.abs(y_true_norm - y_pred_norm))

    # 重心對齊誤差：使用剛剛修改的版本，注意這裡需要傳入 img_sizes
    center_loss = center_alignment_loss(y_true_points, y_pred_points, img_sizes)

    # 每3點群組中心誤差
    center_loss_group = center_alignment_loss_group_of_three(y_true_points, y_pred_points, img_sizes)

    # 返回總 Loss，這裡我們簡單地將兩部分相加（可根據需要調整權重）
    return 0.5 * abs_loss + rel_loss + 0.3 * center_loss



# 從 y_true 中切出前 24 維（代表 12 個點的 (x, y) 座標），然後分別用 MeanAbsoluteError 和 MeanSquaredError 函數來計算預測值和真實值之間的誤差
def custom_mae(y_true, y_pred):
    y_true_points = y_true[:, :24]
    mae_fn = tf.keras.losses.MeanAbsoluteError()
    return mae_fn(y_true_points, y_pred)

def custom_mse(y_true, y_pred):
    y_true_points = y_true[:, :24]
    mse_fn = tf.keras.losses.MeanSquaredError()
    return mse_fn(y_true_points, y_pred)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
              loss=combined_first_loss,
              metrics=[custom_mae, custom_mse])

# Step 4: Train the Model
# Define hyperparameters
batch_size = 16
steps_per_epoch = math.ceil(len(train_filenames) / batch_size)
validation_steps = math.ceil(len(val_filenames) / batch_size)

# 利用 tf.data.Dataset.from_generator() 來包裝 data_generator，並設定好輸出的格式（包括形狀與數據類型），讓 dataset 可以直接傳給模型訓練函數進行訓練
def create_tf_dataset(filenames, batch_size=batch_size, shuffle=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(filenames, batch_size, target_size=(224,224), shuffle=shuffle),
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 26), dtype=tf.float32)
        )
    )
    return dataset

# 生成訓練資料、驗證資料
train_dataset = create_tf_dataset(train_filenames, batch_size=batch_size, shuffle=True)
val_dataset = create_tf_dataset(val_filenames, batch_size=batch_size, shuffle=False)

# Train the model
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# Train the model
print("開始訓練！")
history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=75,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)
print("訓練結束！")

# Step 5: Fine-Tuning (Optional)
# Unfreeze some layers in the base model to fine-tune

for layer in base_model.layers[-3:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

# Recompile with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=combined_loss,  # Smooth L1 loss
              metrics=[custom_mae, custom_mse])
              
# Continue training for a few more epochs
history_fine_tune = model.fit(train_dataset,
                callbacks=[lr_scheduler, early_stopping],
                steps_per_epoch=steps_per_epoch,
                epochs=25,
                validation_data=val_dataset,
                validation_steps=validation_steps,
                verbose=1
                )


# Step 6: Save the Model
model.save("hip_keypoint_model.keras")

# Step 7: Testing and Evaluation
# Plot training history to see loss and metric trends
plt.plot(history.history['loss'], label='Training Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 更改開始(畫出train和val的MAE,MSE圖)
import matplotlib.pyplot as plt

def plot_training_validation_curves(history):
    """
    繪製 Training 與 Validation 的 MAE 與 MSE 曲線。
    """
    epochs = range(1, len(history.history['loss']) + 1)

    # 繪製 MAE 曲線
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # 在 compile() 裡設定的 metrics 產生的鍵名為 custom_mae 與 val_custom_mae
    plt.plot(epochs, history.history['custom_mae'], label='Training MAE')
    plt.plot(epochs, history.history['val_custom_mae'], label='Validation MAE')
    plt.title('Training vs. Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    # 繪製 MSE 曲線
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['custom_mse'], label='Training MSE')
    plt.plot(epochs, history.history['val_custom_mse'], label='Validation MSE')
    plt.title('Training vs. Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_validation_curves(history)
# 更改結束


# 再來才 import cv2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def evaluate_and_record(model, test_filenames, display=True):
    """
    針對測試集中的每張圖片，計算每個關鍵點預測與真實點之間的平均像素誤差，
    並將結果記錄到一個 DataFrame 中；同時在圖片上標記出這個平均誤差（可選）。

    Args:
        model: 已訓練好的模型，輸出 shape 為 (1, 24)（代表 12 個 (x,y) 點）
        test_filenames: 測試集圖片的檔案名稱列表
        display: 如果 True，則對每張圖片顯示預測結果與誤差標記

    Returns:
        pandas.DataFrame: 包含每張圖片名稱與平均像素誤差的表格
    """
    results = []

    for image_name in test_filenames:
        # 讀取圖片及原始尺寸 (例如 (width, height))
        image, original_size = load_image(image_name)  # image shape: (224,224,3)；original_size: (width, height)
        true_labels = load_labels(image_name).reshape(-1, 2)  # 真實標籤 shape: (12, 2)

        # 預測：增加 batch 維度
        input_image = np.expand_dims(image, axis=0)  # shape: (1,224,224,3)
        predictions = model.predict(input_image)     # 預測輸出 shape: (1,24)
        predicted_points = predictions.reshape(-1, 2)  # 轉為 (12,2)

        # 將預測點從標準化尺寸 (224,224) 還原到原始尺寸
        # 假設 original_size = (width, height)
        predicted_points[:, 0] *= original_size[0] / 224
        predicted_points[:, 1] *= original_size[1] / 224

        # 計算每個點的歐氏距離誤差，並取平均值
        errors = np.linalg.norm(predicted_points - true_labels, axis=1)  # (12,)
        avg_error = np.mean(errors)

        # 記錄結果
        results.append({"Image": image_name, "Average Error (px)": avg_error})

        if display:
            # 讀取原始彩色圖片來顯示（因為 load_image 用的是灰階）
            orig_img_path = os.path.join(images_path, image_name)
            original_image = Image.open(orig_img_path).convert("RGB")
            original_image = np.array(original_image)


            # 畫出真實點（綠色）和預測點（紅色）
            for (x, y) in true_labels:
                cv2.circle(original_image, (int(round(x)), int(round(y))), radius=4, color=(0, 255, 0), thickness=-1)
            for (x, y) in predicted_points:
                cv2.circle(original_image, (int(round(x)), int(round(y))), radius=4, color=(255, 0, 0), thickness=-1)

            # 在圖片上標記平均誤差
            cv2.putText(original_image, f"Avg Error: {avg_error:.2f}px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            plt.figure(figsize=(6,6))
            plt.imshow(original_image)
            plt.title(f"{image_name} (Avg Error: {avg_error:.2f}px)")
            plt.axis("off")
            plt.show()

    # 建立表格
    df = pd.DataFrame(results)
    print("Test Set Performance:")
    print(df)
    return df


# 設定固定測試圖片 (也可以用整個測試集)
test_images = [
    "21454344--20230301--Pelvis0.jpg",
    "21884862--20230110--Pelvis0.jpg",
    "21950415--20230131--Pelvis0.jpg",
    "19435927-080.jpg",
    "21921511--20240221--Pelvis0.jpg"
]

# 調用 evaluate_and_record() 來評估模型並生成表格
df_results = evaluate_and_record(model, test_images, display=True)


import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def evaluate_test(model, test_filenames, display=True):
    """
    評估測試集中的所有圖片，計算預測與真實關鍵點的平均像素誤差，
    並以表格方式輸出結果；同時顯示每張圖片並在圖片上標記平均誤差。

    Args:
        model: 已訓練好的模型，輸出 shape 為 (1, 24)（代表 12 個 (x,y) 點）
        test_filenames: 測試集圖片的檔案名稱列表
        display: 如果 True，則對每張圖片顯示預測結果與誤差標記

    Returns:
        pandas.DataFrame: 包含每張圖片名稱與平均像素誤差的表格，以及所有圖片的平均誤差資訊
    """
    results = []

    for image_name in test_filenames:
        # 讀取圖片及原始尺寸 (例如 (width, height))
        image, original_size = load_image(image_name)  # image shape: (224,224,3)；original_size: (width, height)
        true_labels = load_labels(image_name).reshape(-1, 2)  # 真實標籤 shape: (12, 2)

        # 增加 batch 維度以供模型預測
        input_image = np.expand_dims(image, axis=0)  # shape: (1,224,224,3)
        predictions = model.predict(input_image)     # 預測輸出 shape: (1,24)
        predicted_points = predictions.reshape(-1, 2)  # 轉為 (12,2)

        # 將預測點從模型尺寸 (224,224) 還原到原始尺寸
        predicted_points[:, 0] *= original_size[0] / 224
        predicted_points[:, 1] *= original_size[1] / 224

        # 計算每個點的歐氏距離誤差，並取平均值（單位：pixel）
        errors = np.linalg.norm(predicted_points - true_labels, axis=1)  # (12,)
        avg_error = np.mean(errors)

        results.append({
            "Image": image_name,
            "Avg Pixel Error": avg_error
        })

        if display:
            # 讀取原始彩色圖片來顯示（因為 load_image 用的是灰階）
            orig_img_path = os.path.join(images_path, image_name)
            original_image = cv2.imread(orig_img_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # 畫出真實標籤（綠色）和預測標籤（紅色）
            for (x, y) in true_labels:
                cv2.circle(original_image, (int(round(x)), int(round(y))), radius=4, color=(0, 255, 0), thickness=-1)
            for (x, y) in predicted_points:
                cv2.circle(original_image, (int(round(x)), int(round(y))), radius=4, color=(255, 0, 0), thickness=-1)

            # 在圖片上標記平均誤差
            cv2.putText(original_image, f"Avg Error: {avg_error:.2f}px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            plt.figure(figsize=(6,6))
            plt.imshow(original_image)
            plt.title(f"{image_name} (Avg Error: {avg_error:.2f}px)")
            plt.axis("off")
            plt.show()

    # 建立表格並計算整個測試集的平均誤差
    df = pd.DataFrame(results)
    overall_avg_error = df["Avg Pixel Error"].mean()
    print("Test Set Performance:")
    print(df)
    print(f"\nOverall Avg Pixel Error: {overall_avg_error:.2f} pixels")

    return df


df_results = evaluate_test(model, test_filenames, display=True)

# 更改開始(以長條圖畫出每張圖片的 pixel_error)
def plot_average_error_bar(df_results):
    """
    繪製每張測試圖片的平均像素誤差長條圖。
    
    df_results: DataFrame，必須包含 "Image" 與 "Avg Pixel Error" 兩個欄位
    """
    plt.figure(figsize=(14, 6))  # 可以加大圖表寬度，避免標籤擠在一起
    plt.bar(df_results['Image'], df_results['Avg Pixel Error'], color='skyblue')
    plt.title('Average Pixel Error per Test Image')
    plt.xlabel('Image')
    plt.ylabel('Avg Pixel Error (px)')
    plt.xticks(rotation=60, ha='right')  # rotation=60, ha='right' 或 ha='center' 均可嘗試
    plt.tight_layout()
    plt.show()
plot_average_error_bar(df_results)
# 更改結束