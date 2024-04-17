import json
import os.path
import sys
import threading
import time

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, \
    QLabel, QLineEdit, QMessageBox, QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy
from PyQt5.QtGui import QPixmap
from config import Config
from caption import qt_show
from utils import path_checker


class MainWindow(QMainWindow):
    def __init__(self, model_path, word_map_path):
        super().__init__()
        self.setWindowTitle("Image Caption")
        self.setMinimumSize(600, 450)  # 设置窗口的最小大小为 700x700
        self.pixmap = None
        self.model_path = model_path
        dir_path = os.path.dirname(self.model_path)
        log_path = os.path.join(dir_path, "save_log.json")
        log_data = {"save_flag": False}
        with open(log_path, "w") as file:
            json.dump(log_data, file)

        self.log_path = log_path

        self.word_map_path = word_map_path
        central_widget = QWidget()

        self.setCentralWidget(central_widget)

        # 创建布局管理器
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # 输入框与生成文本按钮水平布局
        input_layout = QHBoxLayout()
        main_layout.addLayout(input_layout)

        self.input_path = QLineEdit()
        self.input_path.setText("datasets/img/COCO_test2014_000000000191.jpg")
        input_layout.addWidget(self.input_path)

        self.predict_button = QPushButton('预测')
        input_layout.addWidget(self.predict_button)

        # 保存模型按钮
        self.button = QPushButton('保存模型')
        main_layout.addWidget(self.button)

        self.predict_button.clicked.connect(self.predict_image)
        self.button.clicked.connect(self.toggle_save_flag)

        # 结果标签
        self.result_label = QLabel()
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.result_label)

    def predict_image(self):
        image_path = self.input_path.text()
        if os.path.exists(image_path):
            if os.path.exists(self.model_path):
                qimage = qt_show(self.model_path, image_path, self.word_map_path)
                # 创建 QPixmap 并将 QImage 转换为 QPixmap
                self.pixmap = QPixmap.fromImage(qimage)
                self.result_label.setPixmap(self.pixmap)
                self.result_label.setScaledContents(True)  # 将图片缩放以填充 QLabel
            else:
                QMessageBox.information(self, '提示', '请先保存模型！', QMessageBox.Ok)
        else:
            QMessageBox.information(self, '提示', '图片不存在！', QMessageBox.Ok)

    def toggle_save_flag(self):
        # 将更新后的数据写入 log.json 文件
        log_data = {"save_flag": True}
        with open(self.log_path, "w") as file:
            json.dump(log_data, file)
        self.button.setEnabled(False)  # 设置按钮为不可点击状态
        self.predict_button.setEnabled(False)  # 设置按钮为不可点击状态

        self.save_recall()

    def save_recall(self):
        QMessageBox.information(self, '提示', '模型保存中...', QMessageBox.Ok)
        time.sleep(5)
        save_flag = True
        try:
            with open(self.log_path, "r") as file:
                log_data = json.load(file)
                save_flag = log_data.get("save_flag", True)
        except FileNotFoundError:
            pass

        if not save_flag:
            QMessageBox.information(self, '提示', '模型已保存！', QMessageBox.Ok)
        else:
            QMessageBox.information(self, '提示', '模型保存失败！', QMessageBox.Ok)

        self.predict_button.setEnabled(True)  # 设置按钮为不可点击状态

        # 创建并启动窗口线程
        def button_thread():
            time.sleep(5)
            self.button.setEnabled(True)

        button_thread = threading.Thread(target=button_thread)
        button_thread.start()


if __name__ == '__main__':
    data_folder = f'out_data/coco/out_hdf5/per_5_freq_5_maxlen_100'  # folder with data files saved by create_input_files.py
    data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
    temp_path = 'out_data/coco/save_model'

    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    word_map_file = os.path.normpath(word_map_file)

    app = QApplication(sys.argv)
    filename = 'temp_checkpoint_' + data_name + '.pth'
    model_path = os.path.join(temp_path, filename)
    model_path, _, _ = path_checker(model_path, True, False)
    window = MainWindow(model_path, word_map_file)
    window.show()

    sys.exit(app.exec_())
