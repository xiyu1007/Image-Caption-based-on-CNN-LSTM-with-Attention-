import os.path
import sys
import threading
import time
import matplotlib
import torch
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, \
    QLabel, QLineEdit, QMessageBox, QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy, QRadioButton, QCheckBox, \
    QDoubleSpinBox
from PyQt5.QtGui import QPixmap, QFont, QIcon
from colorama import Fore

from caption import qt_show
from utils import path_checker


class MainWindow(QMainWindow):
    def __init__(self, model_path, word_map_path):
        global pre_image
        super().__init__()
        self.setWindowTitle("Image Caption")
        self.setMinimumSize(800, 650)  # 设置窗口的最小大小为 700x700
        self.pixmap = None
        self.model_path = model_path
        self.save_flag = False
        self.lr_flag = False
        self.main_flag = True
        self.word_map_path = word_map_path
        self.msg_box_saving = QMessageBox()
        self.msg_box_lr = QMessageBox()

        central_widget = QWidget()

        self.setCentralWidget(central_widget)

        # 创建布局管理器
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # 输入框与生成文本按钮水平布局
        pre_layout = QHBoxLayout()
        main_layout.addLayout(pre_layout)
        train_layout = QHBoxLayout()
        main_layout.addLayout(train_layout)
        train_parameter_layout = QHBoxLayout()
        main_layout.addLayout(train_parameter_layout)
        model_layout = QHBoxLayout()
        main_layout.addLayout(model_layout)

        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Enter image path...")
        self.input_path.setText(pre_image)
        pre_layout.addWidget(self.input_path)

        self.button_predict = QPushButton('预测')
        pre_layout.addWidget(self.button_predict)

        # self.is_runing = QRadioButton()
        # self.is_runing.setStyleSheet("""
        #     QRadioButton::indicator:checked {
        #         background-color: green;
        #         border-radius: 7px;
        #     }
        #     QRadioButton::indicator:unchecked {
        #         background-color: gray;
        #         border-radius: 7px;
        #     }
        # """)
        # train_layout.addWidget(self.is_runing)

        # 创建只读文本框
        self.running_train_time = QLineEdit()
        self.running_train_time.setReadOnly(True)  # 设置为只读
        self.running_train_time.setText("模型训练中(回合-批次-时间)：0-0-00:00:00")
        train_layout.addWidget(self.running_train_time)

        self.button_continue = QCheckBox()
        self.button_continue.setEnabled(True)
        self.button_continue.setCheckState(Qt.Checked)
        train_layout.addWidget(self.button_continue)

        # 保存模型按钮
        self.button_save = QPushButton('保存模型')
        self.button_save.setFixedWidth(150)  # 设置固定宽度为200像素
        train_layout.addWidget(self.button_save)

        # 添加一个弹性空间元素，以推动所有控件向右对齐
        train_parameter_layout.addStretch()
        self.label_lr_running = QLabel("Current learning rate")
        self.label_lr_running.setFixedWidth(200)
        train_parameter_layout.addWidget(self.label_lr_running)
        self.text_lr_running = QLineEdit("0.00000")
        self.text_lr_running.setFixedWidth(100)
        train_parameter_layout.addWidget(self.text_lr_running)
        self.label_lr = QLabel("    learning rate")
        self.label_lr.setFixedWidth(150)
        train_parameter_layout.addWidget(self.label_lr)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setFixedWidth(100)
        self.spin_lr.setRange(0.0, 5.0)  # 设置范围为0.0到1.0
        self.spin_lr.setSingleStep(0.01)  # 设置步长为0.01
        self.spin_lr.setValue(0.85)
        train_parameter_layout.addWidget(self.spin_lr)
        self.button_lr = QPushButton()
        self.button_lr.setText("调整学习率")
        self.button_lr.setFixedWidth(150)
        train_parameter_layout.addWidget(self.button_lr)

        self.model_path_text = QLineEdit()
        self.model_path_text.setPlaceholderText("Enter model path...")
        model_layout.addWidget(self.model_path_text)

        # 创建只读文本框
        self.model_train_time = QLineEdit()
        self.model_train_time.setReadOnly(True)  # 设置为只读
        self.model_train_time.setAlignment(Qt.AlignCenter)  # 设置文本居中显示
        self.model_train_time.setText("0-0-00:00:00")
        self.model_train_time.setFixedWidth(200)  # 设置固定宽度为200像素
        # self.model_train_time.setFixedWidth(self.model_train_time.sizeHint().width()+100)
        model_layout.addWidget(self.model_train_time)

        self.button_predict.clicked.connect(self.predict_image)
        self.button_continue.clicked.connect(self.training_continue)
        self.button_save.clicked.connect(self.toggle_save_flag)
        self.button_lr.clicked.connect(self.toggle_lr_flag)
        self.model_path_text.textChanged.connect(self.text_changed_slot)

        # 结果标签
        self.result_label = QLabel()
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.result_label)

    def predict_image(self):
        image_path = self.input_path.text()

        if os.path.exists(image_path):
            self.ban_button()
            self.button_predict.setText("预测中...")
            model_path_text = self.model_path_text.text()

            def show_thread(model_path):
                qimage = qt_show(model_path, image_path, self.word_map_path)
                # 创建 QPixmap 并将 QImage 转换为 QPixmap
                self.pixmap = QPixmap.fromImage(qimage)
                self.result_label.setPixmap(self.pixmap)
                self.result_label.setScaledContents(True)  # 将图片缩放以填充 QLabel
                self.enable_button()

            if model_path_text == "":
                show_t = threading.Thread(target=show_thread, args=(self.model_path,))
                show_t.start()
            elif not os.path.exists(model_path_text):
                self.enable_button()
                QMessageBox.information(self, '提示', '模型不存在！', QMessageBox.Ok)
            else:
                show_t = threading.Thread(target=show_thread, args=(model_path_text,))
                show_t.start()
        else:
            QMessageBox.information(self, '提示', '图片不存在！', QMessageBox.Ok)

    def toggle_lr_flag(self):
        self.lr_flag = True
        self.save_flag = False
        self.ban_button()
        print(self.text_lr_running.text())
        print(float(self.text_lr_running.text()))
        current_lr = float(self.text_lr_running.text()) * float(window.spin_lr.text())

        self.text_lr_running.setText(str(format(current_lr, '.3e')))

        self.msg_box_lr.setText('学习率已调整！')
        # 显示模型保存中提示
        self.msg_box_lr.exec_()
        QTimer.singleShot(5000, self.enable_button)

    def toggle_save_flag(self):
        self.save_flag = True
        self.ban_button()
        self.msg_box_saving.setText('模型保存中...')
        # 显示模型保存中提示
        self.msg_box_saving.exec_()

    def get_continue_flag(self):
        return self.button_continue.isChecked()

    def training_continue(self):
        self.ban_button()
        QTimer.singleShot(500, self.enable_button)
        if not self.button_continue.isChecked():
            msg_box_continue = QMessageBox()
            # msg_box_continue.setStyleSheet("QLabel { text-align: center; }")
            msg_box_continue.setWindowTitle("警告")
            msg_box_continue.setText('取消选中，保存模型后将停止训练！  ')
            msg_box_continue.exec_()

    def ban_button(self):
        self.button_save.setEnabled(False)  # 设置按钮为不可点击状态
        self.button_lr.setEnabled(False)  # 设置按钮为不可点击状态
        self.spin_lr.setEnabled(False)  # 设置按钮为不可点击状态
        self.button_continue.setEnabled(False)  # 设置按钮为不可点击状态
        self.button_predict.setEnabled(False)  # 设置按钮为不可点击状态
        self.button_predict.setText("刷新中...")  # 设置按钮为不可点击状态

    def enable_button(self):
        try:
            self.button_save.setEnabled(self.main_flag)
            self.button_continue.setEnabled(self.main_flag)  # 设置按钮为不可点击状态
            self.button_lr.setEnabled(self.main_flag)  # 设置按钮为不可点击状态
            self.spin_lr.setEnabled(self.main_flag)  # 设置按钮为不可点击状态
            self.button_predict.setText("预测")  # 设置按钮为不可点击状态
            self.button_predict.setEnabled(True)  # 设置按钮为不可点击状态
        except Exception as e:
            self.close()
            print(Fore.YELLOW + "Error enable_button: ", e)

    def save_recall(self, msg_box_saving):
        try:
            if not self.save_flag:
                msg_box_saving.setText('模型已保存！')
            else:
                msg_box_saving.setText('模型保存失败！')
        except Exception as e:
            msg_box_saving.setText('出错，模型保存失败！')
            print(Fore.YELLOW + "\nsave_recall：", e)
        msg_box_saving.exec_()

        def button_reset():
            time.sleep(3)
            self.enable_button()

        try:
            show_t = threading.Thread(target=button_reset)
            show_t.start()
        except Exception as e:
            print(Fore.YELLOW + "\nsave_recall--button_reset：", e)
            self.enable_button()

    def text_changed_slot(self, checkpoint):
        def text_thread(checkpoint):
            try:
                checkpoint = torch.load(checkpoint)
                train_time = checkpoint['train_time']
                number = checkpoint['number']
                epoch = checkpoint['epoch']
                self.model_train_time.setText(f"{epoch}-{str(number)}-{train_time}")
                # self.model_train_time.setFixedWidth(self.model_train_time.sizeHint().width() + 100)
            except Exception as e:
                print(Fore.YELLOW + "\nError loading model:", e)
            self.model_path_text.textChanged.connect(self.text_changed_slot)

        if os.path.exists(checkpoint):
            self.model_path_text.textChanged.disconnect(self.text_changed_slot)
            set_text = threading.Thread(target=text_thread, args=(checkpoint,))
            set_text.start()

    def set_train_time(self, train_time, number, epoch, lr):
        try:
            self.running_train_time.setText(f"模型训练中(回合-批次-时间)：{epoch}-{str(number)}-{train_time}")
            self.text_lr_running.setText(str(format(lr, '.3e')))
            return True
        except Exception as e:
            print(Fore.YELLOW + "\nError set_train_time:", e)
            return False


pre_image = 'datasets/img/img.png'

if __name__ == '__main__':
    # TODO 请确保由creat_input_files.py生成的word_map_file 路径与需要查看的模型匹配
    # matplotlib.use('TkAgg')  # 多线程错误问题,请使用Agg

    datasets_name = 'coco_no_premodel'
    data_folder = f'out_data/{datasets_name}/out_hdf5/per_5_freq_1_maxlen_50'  # folder with data files saved by create_input_files.py
    data_name = f'{datasets_name}_5_cap_per_img_1_min_word_freq'  # base name shared by data files
    temp_path = f'out_data/{datasets_name}/save_model'

    checkpoint = r"out_data/coco/save_model/temp_checkpoint_coco_5_cap_per_img_5_min_word_freq_3318.pth"
    checkpoint, _, _ = path_checker(checkpoint, True, False)

    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    word_map_file = os.path.normpath(word_map_file)

    app = QApplication(sys.argv)

    # 设置应用程序的窗口图标
    app.setWindowIcon(QIcon('utils/main.jpg'))

    # 创建 QFont 实例并设置字体和字号
    font = QFont()
    font.setFamily("Arial")  # 设置字体
    font.setPointSize(12)  # 设置字号

    # 将 QFont 应用到应用程序上的所有 QLineEdit 控件
    app.setFont(font, "QLineEdit")
    app.setFont(font, "QPushButton")
    app.setFont(font, "QMessageBox")
    app.setFont(font, "QDoubleSpinBox")
    app.setFont(font, "QLabel")

    filename = 'checkpoint_' + data_name + '_epoch_2' + '.pth'
    model_path = os.path.join(temp_path, filename)
    model_path, _, _ = path_checker(model_path, True, False)
    window = MainWindow(model_path, word_map_file)
    window.main_flag = False
    window.ban_button()
    window.enable_button()
    window.button_continue.setCheckState(Qt.Unchecked)
    window.running_train_time.setText("无训练中的模型(回合-批次-时间)：0-0-00:00:00")

    window.show()

    sys.exit(app.exec_())
