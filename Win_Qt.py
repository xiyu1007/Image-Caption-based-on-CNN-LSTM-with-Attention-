import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox
from config import save_flag

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("参数设置")
        self.setGeometry(100, 100, 300, 200)
        
        self.button = QPushButton('保存模型', self)
        self.button.setGeometry(50, 50, 200, 50)
        self.button.clicked.connect(self.toggle_save_flag)
        
    def toggle_save_flag(self):
        global save_flag
        save_flag = not save_flag
        QMessageBox.information(self, '提示', '模型已保存！', QMessageBox.Ok)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())