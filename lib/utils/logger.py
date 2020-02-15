import time
import os
import shutil


class Timer(object):
    def __init__(self):
        self.start_time = 0
        self.iter_length = 0

    def start(self, iter_length):
        self.iter_length = iter_length
        self.start_time = time.time()

    def stamp(self, step):
        time_duration = time.time() - self.start_time
        rest_time = time_duration / (step + 1) * (self.iter_length - step - 1)
        cur_hour, cur_min, cur_sec = self.convert_format(time_duration)
        rest_hour, rest_min, rest_sec = self.convert_format(rest_time)
        log_string = "[{}:{}:{} < {}:{}:{}]".format(cur_hour, cur_min, cur_sec, rest_hour, rest_min, rest_sec)
        return log_string

    @staticmethod
    def convert_format(sec):
        hour = "{:02}".format(int(sec // 3600))
        minu = "{:02}".format(int((sec % 3600) // 60))
        sec = "{:02}".format(int(sec % 60))
        return hour, minu, sec


class Logger(object):
    def __init__(self, iter_length, print_interval=50):
        self.timer = Timer()
        self.cur_step = 0
        self.print_interval = print_interval
        self.total_loss = 0
        self.template = "{} | Iter. {}  Loss: {}"
        self.timer.start(iter_length)
        self.log_path = "./log"

    def write_log_file(self, text):
        with open(os.path.join(self.log_path, "log.txt"), "a+") as writer:
            writer.write(text + "\n")

    def step(self, loss, step):
        self.total_loss += loss
        self.cur_step += 1
        if self.cur_step == self.print_interval:
            line = self.template.format(self.timer.stamp(step), step, self.total_loss / self.cur_step)
            print(line)
            self.write_log_file(line)
            self.cur_step = 0
            self.total_loss = 0
