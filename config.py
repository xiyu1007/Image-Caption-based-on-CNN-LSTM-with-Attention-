# config.py
import time

# 初始化参数为False
save_flag = False

# 设置超时时间为1个小时
timeout = 1 * 60 * 60

# 记录上次保存的时间
last_save_time = time.time()

# 检查是否超过超时时间，如果是，则将保存参数设置为True
def check_timeout():
    global save_flag, last_save_time
    current_time = time.time()
    if current_time - last_save_time > timeout:
        save_flag = True
        last_save_time = current_time