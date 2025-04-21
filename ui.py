# hip_ui.py
# 基本 UI 架構：圖片載入 + 模型推論 + 點位顯示 + SQLite 儲存（骨架版）

import ctypes
import sys
import os
import sqlite3
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout,
                             QWidget, QHBoxLayout, QMessageBox)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QFont
from PyQt5.QtCore import Qt, QPoint, QSize
import numpy as np
import tensorflow as tf
from PIL import Image

DB_PATH = "hip_records.db"
MODEL_PATH = "models\model_e1000_bs16_lr0.01_p50_efficientnetv2m.keras"

MAX_DISPLAY_SIZE = QSize(700, 700)  # 圖片顯示最大寬高

class ImageCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.points = []  # (x, y)

    def set_image(self, path):
        self.img_path = path
        self.pixmap_orig = QPixmap(path).scaled(MAX_DISPLAY_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(self.pixmap_orig)
        self.setFixedSize(self.pixmap_orig.size())  # 確保 QLabel 大小與圖片一致
        self.points.clear()
        self.repaint()

    def set_points(self, points):
        self.points = points
        self.repaint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            self.points.append((pos.x(), pos.y()))
            self.repaint()

    def paintEvent(self, event):
        super().paintEvent(event)
        if hasattr(self, 'pixmap_orig') and self.points:
            painter = QPainter(self)
            pen = QPen(QColor(255, 0, 0), 6)
            painter.setPen(pen)
            font = QFont()
            font.setPointSize(10)
            painter.setFont(font)
            for i, (x, y) in enumerate(self.points):
                painter.drawPoint(int(x), int(y))
                painter.drawText(int(x) + 5, int(y) - 5, str(i + 1))

class HipApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hip X-ray Annotation Tool")
        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        self.init_ui()
        self.init_db()

    def init_ui(self):
        self.canvas = ImageCanvas()

        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)

        self.btn_predict = QPushButton("Run Model")
        self.btn_predict.clicked.connect(self.run_model)

        self.btn_save = QPushButton("Save to DB")
        self.btn_save.clicked.connect(self.save_record)

        self.btn_clear = QPushButton("Clear Points")
        self.btn_clear.clicked.connect(self.clear_points)

        hbox = QHBoxLayout()
        hbox.addWidget(self.btn_load)
        hbox.addWidget(self.btn_predict)
        hbox.addWidget(self.btn_save)
        hbox.addWidget(self.btn_clear)

        layout = QVBoxLayout()
        layout.addLayout(hbox)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def init_db(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS records (
                        id INTEGER PRIMARY KEY,
                        img_path TEXT,
                        keypoints TEXT
                    )''')
        conn.commit()
        conn.close()

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Image', '.', 'Image files (*.jpg *.png)')
        if fname:
            self.canvas.set_image(fname)

    def run_model(self):
        if not hasattr(self.canvas, 'img_path'):
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        # 讀取圖片並前處理
        img = Image.open(self.canvas.img_path).convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = self.model.predict(img_array)[0]  # shape: (24,) x1,y1,x2,y2...
        print("模型預測點（前幾個）:", preds[:6])

        # 模型輸出是 pixel 座標（相對於 224x224），需轉成顯示比例
        scale_x = self.canvas.width() / 224
        scale_y = self.canvas.height() / 224
        points = [(preds[i] * scale_x, preds[i+1] * scale_y) for i in range(0, 24, 2)]

        self.canvas.set_points(points)
        self.points = points

    def save_record(self):
        if not hasattr(self.canvas, 'img_path') or not hasattr(self, 'points'):
            QMessageBox.warning(self, "Warning", "Missing image or keypoints.")
            return
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO records (img_path, keypoints) VALUES (?, ?)",
                  (self.canvas.img_path, str(self.points)))
        conn.commit()
        conn.close()
        QMessageBox.information(self, "Saved", "Record saved to database.")

    def clear_points(self):
        self.canvas.set_points([])
        self.points = []

if __name__ == '__main__':
    dll_path = os.path.abspath("tf_gpu_env/Lib/site-packages/cv2/zlibwapi.dll")
    ctypes.WinDLL(dll_path)
    print("DLL loaded from:", dll_path)
    app = QApplication(sys.argv)
    window = HipApp()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
