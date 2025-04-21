#2025/3/11 修改loss function部分，以及第一次訓練次數下調為25
#2025/3/11 修改預測圖片的部分，固定預測某5張圖片，方便之後比較模型預測效果

# Step 1: Setup Google Colab environment and install necessary libraries
# Install TensorFlow and EfficientNet libraries
# !pip install tensorflow
# !pip install -q efficientnet

# Import the required packages
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import efficientnet.tfkeras as efn
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import random
import math
from sklearn.model_selection import train_test_split

print("Available GPUs:", tf.config.experimental.list_physical_devices('GPU'))


# Step 2: Data Preprocessing
# Load dataset, which consists of images and the coordinates of 12 key points for the hip

resolution = 224

images_path = "./images/"
labels_folder_path = "./annotations/"

from tensorflow.keras import layers, Model
# 更改開始
# 建構模型
def build_model():
    inputs = layers.Input(shape=(224, 224, 3), name="image_input")
    x = layers.Conv2D(32, (3,3), activation="relu")(inputs)
    x = layers.Flatten()(x)
    outputs = layers.Dense(24, activation="linear")(x)  # 輸出 24 維
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_model()
model.summary()
# 更改結束

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
train_filenames, val_filenames = train_test_split(image_filenames, test_size=0.15, random_state=42)

# Load and preprocess an image
def load_image(image_name, target_size=(resolution, resolution)):
    image_path = os.path.join(images_path, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_size = (image.shape[1], image.shape[0])
    image = cv2.resize(image, target_size) / 255.0
    image = np.stack([image] * 3, axis=-1)  # Normalize to [0, 1]
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
    center_x = target_size[1] / 2
    center_y = target_size[0] / 2


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
            
            # 更改開始
            # 強制 shape 為 (1,24)
            labels = np.array(labels, dtype=np.float32).reshape(1, 24)
            # 強制原始尺寸為 (1,2)
            original_size = np.array(original_size, dtype=np.float32).reshape(1, 2)
            # 合併成 (1,26)
            labels_with_size = np.concatenate([labels, original_size], axis=1)
            
            batch_images.append(image)
            batch_labels.append(labels_with_size)
            # 更改結束
            if len(batch_images) == batch_size:
                # 更改開始
                batch_images = np.array(batch_images, dtype=np.float32)
                batch_labels = np.array(batch_labels, dtype=np.float32).reshape(-1, 26)
                
                yield batch_images, batch_labels
                batch_images, batch_labels = [], []
                # 更改結束
        if len(batch_images) > 0:
            # 更改開始
            batch_images = np.array(batch_images, dtype=np.float32)
            batch_labels = np.array(batch_labels, dtype=np.float32).reshape(-1, 26)
            
            yield batch_images, batch_labels
            batch_images, batch_labels = [], []
            # 更改結束




# Step 3: Model Definition
# Load EfficientNetB0 model and modify the output for keypoint regression
base_model = efn.EfficientNetB0(input_shape=(resolution, resolution, 3), include_top=False, weights='imagenet')
base_model.trainable = True

# 更改開始
# 自訂分類：先 Global Average Pooling，再一個 2048 維的全連接層（ReLU），再輸出 24 維
global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
dense_1 = tf.keras.layers.Dense(2048, activation='relu')(global_avg_pool)
output_layer = tf.keras.layers.Dense(24, activation='linear')(dense_1)
# 更改結束

# 組合模型：輸入依然是 EfficientNet 的輸入，輸出為新的 24 維分類頭
model = tf.keras.models.Model(inputs=base_model.input, outputs=output_layer)

# 更改開始
# loss function
def huber_loss(y_true, y_pred):
    """ 計算單點位置的 Huber Loss """
    return tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)


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

    # 返回總 Loss，這裡我們簡單地將兩部分相加（可根據需要調整權重）
    return abs_loss + rel_loss


@tf.function
def combined_loss(y_true, y_pred):
    """
    綜合 Loss：使用 Huber Loss 衡量預測點與真實點之間的絕對位置誤差，
    並使用 L1 Loss 衡量預測點相對於圖片尺寸的比例誤差。

    y_true shape: (batch_size, 26) -> 前 24 維為關鍵點，後 2 維為圖片原始尺寸
    y_pred shape: (batch_size, 24) -> 預測 12 個點
    """
    # 強制轉換為 float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # 從 y_true 拆分出圖片尺寸和真實點
    img_sizes = y_true[:, -2:]     # (batch_size, 2)
    y_true_points = y_true[:, :-2]  # (batch_size, 24)

    # Reshape 為 (batch_size, 12, 2)
    y_true_points = tf.reshape(y_true_points, (-1, 12, 2))
    y_pred_points = tf.reshape(y_pred, (-1, 12, 2))

    # 計算 Huber Loss
    huber = tf.keras.losses.Huber(delta=1.0)(y_true_points, y_pred_points)

    # 將點數據歸一化到相對於圖片尺寸的比例，並計算 L1 Loss
    y_true_norm = y_true_points / img_sizes[:, None, :]
    y_pred_norm = y_pred_points / img_sizes[:, None, :]
    relative_loss = tf.reduce_mean(tf.abs(y_true_norm - y_pred_norm))

    return huber + relative_loss



# 從 y_true 中切出前 24 維（代表 12 個點的 (x, y) 座標），然後分別用 MeanAbsoluteError 和 MeanSquaredError 函數來計算預測值和真實值之間的誤差
def custom_mae(y_true, y_pred):
    y_true_points = y_true[:, :24]
    mae_fn = tf.keras.losses.MeanAbsoluteError()
    return mae_fn(y_true_points, y_pred)

def custom_mse(y_true, y_pred):
    y_true_points = y_true[:, :24]
    mse_fn = tf.keras.losses.MeanSquaredError()
    return mse_fn(y_true_points, y_pred)
# 更改結束

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
              # 更改開始
              loss=combined_first_loss,  
              metrics=[custom_mae, custom_mse])
              # 更改結束

# Step 4: Train the Model
# Define hyperparameters
batch_size = 8
steps_per_epoch = math.ceil(len(train_filenames) / batch_size)
validation_steps = math.ceil(len(val_filenames) / batch_size)

# 更改開始
# 利用 tf.data.Dataset.from_generator() 來包裝 data_generator，並設定好輸出的格式（包括形狀與數據類型），讓 dataset 可以直接傳給模型訓練函數進行訓練
def create_tf_dataset(filenames, batch_size=8):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(filenames, batch_size, target_size=(224,224)),
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 26), dtype=tf.float32)
        )
    )
    return dataset

# 生成訓練資料、驗證資料
train_dataset = create_tf_dataset(train_filenames, batch_size=8)
val_dataset = create_tf_dataset(val_filenames, batch_size=8)
# 更改結束

# Train the model
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)


# 更改開始
# Train the model
history = model.fit(
    train_dataset,
    steps_per_epoch=math.ceil(len(train_filenames)/8),
    epochs=25,
    validation_data=val_dataset,
    validation_steps=math.ceil(len(val_filenames)/8),
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)
# 更改結束

# Step 5: Fine-Tuning (Optional)
# Unfreeze some layers in the base model to fine-tune
# 更改開始
for layer in base_model.layers[-3:]:# 更改結束
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

# Recompile with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              # 更改開始
              loss=combined_loss,  # Smooth L1 loss
              metrics=[custom_mae, custom_mse])
              # 更改結束

# 更改開始
# Continue training for a few more epochs
history_fine_tune = model.fit(train_dataset,
                callbacks=[lr_scheduler, early_stopping],
                steps_per_epoch=math.ceil(len(train_filenames)/8),
                epochs=20,
                validation_data=val_dataset,
                validation_steps=math.ceil(len(val_filenames)/8),
                verbose=1
                )
# 更改結束


# Step 6: Save the Model
model.save("hip_keypoint_model.keras")

# Step 7: Testing and Evaluation
# Write testing code here if you have a test dataset available

# Plot training history to see loss and metric trends
plt.plot(history.history['loss'], label='Training Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 更改開始
def visualize_predictions(model, fixed_image_filenames):
    # fixed_image_filenames 是你指定的圖片檔名列表
    for i, image_name in enumerate(fixed_image_filenames):
        # 讀取圖片與原始尺寸
        image, original_size = load_image(image_name)
        # 讀取真實標籤（轉成 (N,2) 格式）
        labels = load_labels(image_name).reshape(-1, 2)
        # 準備模型推論（增加 batch 維度）
        input_image = np.expand_dims(image, axis=0)
        predictions = model.predict(input_image)  # 預測輸出 shape: (1, 24)
        predicted_points = predictions.reshape(-1, 2)

        # 將預測點縮放回原圖尺寸
        predicted_points[:, 0] *= original_size[0] / 224
        predicted_points[:, 1] *= original_size[1] / 224

        # 讀取原始彩色圖片
        original_image = cv2.imread(os.path.join(images_path, image_name))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # 在圖片上畫出真實標籤（綠色）
        for (x, y) in labels:
            cv2.circle(original_image, (int(round(x)), int(round(y))), radius=4, color=(0,255,0), thickness=-1)
        # 在圖片上畫出模型預測（紅色）
        for (x, y) in predicted_points:
            cv2.circle(original_image, (int(round(x)), int(round(y))), radius=4, color=(255,0,0), thickness=-1)

        # 顯示圖片
        plt.figure(figsize=(6,6))
        plt.imshow(original_image)
        plt.title(f"Sample {i+1}: {image_name}")
        plt.axis("off")
        plt.show()

# 固定圖片檔名列表
fixed_images = [
    "21454344--20230301--Pelvis0.jpg",
    "21884862--20230110--Pelvis0.jpg",
    "21950415--20230131--Pelvis0.jpg",
    "19435927-080.jpg",
    "21921511--20240221--Pelvis0.jpg"
]

visualize_predictions(model, fixed_images)