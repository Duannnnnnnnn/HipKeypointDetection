# predict.py for Keras-trained model
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetV2S, EfficientNetV2M, EfficientNetB0, EfficientNetB1, EfficientNetB2
from train5 import combined_first_loss


IMAGE_SIZE = 224
POINTS_COUNT = 12

def build_model(model_name):
    if model_name == "efficientnetv2s":
        base = EfficientNetV2S(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    elif model_name == "efficientnetv2m":
        base = EfficientNetV2M(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    elif model_name == "efficientnetb0":
        base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    elif model_name == "efficientnetb1":
        base = EfficientNetB1(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    elif model_name == "efficientnetb2":
        base = EfficientNetB2(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    else:
        raise ValueError("Unsupported model name")

    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    outputs = tf.keras.layers.Dense(24, activation='linear')(x)
    return tf.keras.Model(inputs=base.input, outputs=outputs)

def load_annotations(annotation_path):
    keypoints = pd.read_csv(annotation_path, header=None).values.flatten()
    keypoints = [float(coord) for point in keypoints for coord in point.strip("()\"").split(",")]
    return np.array(keypoints).reshape(-1, 2)

def calculate_avg_distance(predicted, original):
    return np.mean(np.linalg.norm(predicted - original, axis=1))

def predict(model_path, data_dir, output_dir):
    # Load model
    model = load_model(model_path)
    model_name = os.path.basename(model_path).split("_")[-1].replace(".keras", "")
    result_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(model_path))[0])
    os.makedirs(result_dir, exist_ok=True)

    ranges = {"0-30": [], "31-60": [], "61-90": [], "91+": []}
    for r in ranges: os.makedirs(os.path.join(result_dir, r), exist_ok=True)

    image_files = sorted(os.listdir(os.path.join(data_dir, 'images')))
    distances = []

    for idx, image_file in enumerate(image_files):
        if not image_file.endswith(".jpg"): continue
        image_path = os.path.join(data_dir, 'images', image_file)
        annotation_path = os.path.join(data_dir, 'annotations', image_file.replace(".jpg", ".csv"))

        image = Image.open(image_path).convert("L")
        original_size = image.size
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.array(image, dtype=np.float32) / 255.
        arr = np.stack([arr]*3, axis=-1)
        input_tensor = np.expand_dims(arr, axis=0)

        prediction = model.predict(input_tensor)[0].reshape(-1, 2)
        prediction[:, 0] *= original_size[0] / IMAGE_SIZE
        prediction[:, 1] *= original_size[1] / IMAGE_SIZE

        original = load_annotations(annotation_path)
        avg_dist = calculate_avg_distance(prediction, original)
        distances.append(avg_dist)

        if avg_dist <= 30:
            folder = "0-30"
        elif avg_dist <= 60:
            folder = "31-60"
        elif avg_dist <= 90:
            folder = "61-90"
        else:
            folder = "91+"

        # Plot
        plt.imshow(Image.open(image_path).convert("RGB"))
        plt.scatter(prediction[:, 0], prediction[:, 1], c='yellow', label='Predicted', s=10)
        plt.scatter(original[:, 0], original[:, 1], c='red', label='Original', s=10)
        plt.title(f"{image_file} | Dist: {avg_dist:.2f}")
        plt.axis('off')
        plt.legend()
        plt.savefig(os.path.join(result_dir, folder, image_file.replace(".jpg", "_pred.png")))
        plt.close()

        np.savetxt(os.path.join(result_dir, folder, image_file.replace(".jpg", "_keypoints.txt")), prediction, fmt="%.2f", delimiter=",")

    # Summary chart
    plt.figure(figsize=(14, 5))
    plt.bar(range(len(distances)), distances, color='blue')
    avg_all = np.mean(distances)
    plt.axhline(avg_all, color='red', linestyle='--', label=f'Overall Avg: {avg_all:.2f}')
    plt.xlabel('Image Index')
    plt.ylabel('Avg Pixel Error')
    plt.title('Prediction Error per Image')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{model_name}_avg_distances.png"))
    plt.show()
    print(f"✅ Overall average pixel error: {avg_all:.2f} px")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    predict(args.model_path, args.data, args.output_dir)
