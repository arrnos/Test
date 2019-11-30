#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: batch_process_file.py
@time: 2019/11/28
"""

import os
import threading
import time

class my_thread(threading.Thread):
    def __init__(self, name, in_file, out_file, func):
        threading.Thread.__init__(self)
        self.name = name
        self.in_file = in_file
        self.out_file = out_file
        self.func = func

    def run(self):
        print("开始线程：" + self.name)
        self.func(self.in_file, self.out_file)
        print("退出线程：" + self.name)


def batch_process_file(process_func, input_file_path, output_file_path, n_threads=25):
    time1 = time.time()
    assert input_file_path, "输入文件不存在！"
    file_name_base = os.path.basename(input_file_path) + "_"
    file_dir = os.path.dirname(input_file_path)
    print(os.popen("wc -l %s" % input_file_path).readlines())
    file_line_num = int(os.popen("wc -l %s" % input_file_path).readlines()[0].strip().split(" ")[0])
    print("输入文件总行数：", file_line_num)
    avg_line_num = file_line_num // n_threads + 1
    cmd = "split -l %d %s %s" % (avg_line_num, input_file_path, file_name_base)
    print(cmd)
    os.chdir(os.path.dirname(input_file_path))
    os.system(cmd)

    split_input_file_ls = sorted([os.path.join(file_dir, x) for x in os.listdir(file_dir) if file_name_base in x])
    split_output_file_ls = [x + "_out" for x in split_input_file_ls]
    print("split_input_file:\n", "\n".join(split_output_file_ls))

    thread_ls = []
    for i, (input_file_i, output_file_i) in enumerate(zip(split_input_file_ls, split_output_file_ls)):
        thread_i = my_thread(str(i), input_file_i, output_file_i, process_func)
        thread_i.start()
        thread_ls.append(thread_i)

    for thread in thread_ls:
        thread.join()

    cmd_cat = "cat %s > %s" % (" ".join(split_output_file_ls), output_file_path)
    print(cmd_cat)
    os.system(cmd_cat)

    cmd_remove = "rm %s" % (" ".join(split_input_file_ls + split_output_file_ls))
    print(cmd_remove)
    os.system(cmd_remove)

    out_file_line_num = int(os.popen("wc -l %s" % output_file_path).readlines()[0].strip().split(" ")[0])
    time2 = time.time()
    print("处理任务已完成！耗时：%.3f min,输出文件行数：%d, outfile:%s" % ((time2-time1)/60,out_file_line_num,output_file_path))



def process_func_test(in_file, out_file):
    import codecs
    with codecs.open(in_file, 'r', 'utf-8') as fin, codecs.open(out_file, 'w', 'utf-8') as fout:
        for line in fin:
            if line:
                fout.write(line[:30] + "\n")
            else:
                fout.write(line)


if __name__ == '__main__':
    from config.file_path_config import *
    import sys
    n_threads = int(sys.argv[1])
    batch_process_file(process_func_test, test_raw_data_file, "./out_test", n_threads)
