# config.py
import time


class Config:
    def __init__(self):
        # 初始化参数为False
        self.save_flag = False

        # 设置超时时间为1个小时
        self.timeout = 1 * 60 * 60

        # 记录上次保存的时间
        self.last_save_time = time.time()

        # 检查是否超过超时时间，如果是，则将保存参数设置为True

    def check_timeout(self):
        current_time = time.time()
        if current_time - self.last_save_time > self.timeout:
            self.save_flag = True
            self.last_save_time = current_time


if __name__ == '__main__':
    pass
