import threading
import os
import csv
import Queue
import sys
import numpy as np
import time
from multiprocessing import Pool


states_num_one_line = 13
labels_num_one_line = 4

input_side_dimension = 64
img_wid = input_side_dimension
img_height = input_side_dimension
lock = threading.Lock()

data_read_flags = [False, False, False, False]
data_house = [0, 0, 0, 0]
isAllDataRead = False


path = "/home/lucasyu/YU/backward_unable"
states_filename = ["chg_route1_trial3_swinging/uav_data_2018_12_06.csv"]#,
                    # "hzy_route1_trial1/uav_data_2018_12_06.csv",
                    # "hzy_route1_trial2/uav_data_2018_12_06.csv"]
                   # "chg_route1_trial1/uav_data_2018_12_06.csv"]
labels_filename = ["chg_route1_trial3_swinging/label_data_2018_12_06.csv"]#,
                    # "hzy_route1_trial1/label_data_2018_12_06.csv",
                    # "hzy_route1_trial2/label_data_2018_12_06.csv"]
                   # "chg_route1_trial1/label_data_2018_12_06.csv"]
clouds_filename = ["chg_route1_trial3_swinging/pcl_data_2018_12_06.csv"]#,
                    # "hzy_route1_trial1/pcl_data_2018_12_06.csv",
                    # "hzy_route1_trial2/pcl_data_2018_12_06.csv"]
                   # "chg_route1_trial1/pcl_data_2018_12_06.csv"]

file_path_list_states = [os.path.join(path, states) for states in states_filename]
file_path_list_pcl = [os.path.join(path, cloud) for cloud in clouds_filename]
file_path_list_labels = [os.path.join(path, label) for label in labels_filename]


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result


def csv_read(queuq_filenames):
    filename = queuq_filenames.get()
    with open(filename, mode='r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        i_row = 0
        for row in csv_reader:
            print row[0]


def read_all_three_csvs(filename_pcl, filename_states,
                        num_one_line_states, filename_labels, num_one_line_labels):
    maxInt = sys.maxsize
    decrement = True
    while decrement:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        decrement = False
        try:
            print "begin read pcl data.."
            csv.field_size_limit(maxInt)

            file = open(filename_states, "r")
            num_lines = len(file.readlines())
            file.close()
            data_states = np.zeros([num_lines, num_one_line_states])
            with open(filename_states, mode='r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                i_row = 0
                for row in csv_reader:
                    for i in range(num_one_line_states):
                        data_states[i_row, i] = row[i]
                    i_row = i_row + 1

            file = open(filename_labels, "r")
            num_lines = len(file.readlines())
            file.close()
            data_labels = np.zeros([num_lines, num_one_line_labels])
            with open(filename_labels, mode='r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                i_row = 0
                for row in csv_reader:
                    for i in range(num_one_line_labels):
                        data_labels[i_row, i] = row[i]
                    i_row = i_row + 1

            clouds = open(filename_pcl, "r")
            img_num = len(clouds.readlines())
            clouds.close()
            data_pcl = np.ones([img_num, img_wid, img_wid, img_height, 1])
            with open(filename_pcl, mode='r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                i_row = 0
                for row in csv_reader:
                    for i in range(img_wid):
                        for j in range(img_wid):
                            for k in range(img_height):
                                data_pcl[i_row, i, j, k, 0] = row[i * img_wid + j * img_wid + k * img_height]
                    i_row = i_row + 1
                # list_result.append(data)
        except OverflowError:
            maxInt = int(maxInt / 10)
            decrement = True
    global data_house
    global data_read_flags
    for i_pool in range(4):
        if not (data_read_flags[i_pool]):
            data_house[i_pool] = [data_pcl, data_states, data_labels]
            data_read_flags[i_pool] = True
            print "data is already put into the data house.."
            break
    return True
    # return data_pcl, data_states, data_labels


def monitor_and_get_data():
    global data_read_flags
    isLookingForData = True
    print "looking for data.."
    while isLookingForData:
        for i_flag in range(len(data_read_flags)):
            if data_read_flags[i_flag]:
                [data_pcl, data_states, data_labels] = data_house[i_flag]
                data_read_flags[i_flag] = False
                isLookingForData = False
                print "get data: ", data_pcl[0], data_states[0], data_labels[0]
                break
    return True
    # TODO: shuffle datas and train


def read_others(q_file, num_one_line):
    maxInt = sys.maxsize
    decrement = True
    filename = q_file.get()
    while decrement:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        decrement = False
        try:
            csv.field_size_limit(maxInt)
            file = open(filename, "r")
            num_lines = len(file.readlines())
            file.close()
            data = np.zeros([num_lines, num_one_line])
            with open(filename, mode='r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                i_row = 0
                for row in csv_reader:
                    for i in range(num_one_line):
                        data[i_row, i] = row[i]
                    i_row = i_row + 1
        except OverflowError:
            maxInt = int(maxInt / 10)
            decrement = True

    return data


def enqueue_file(queue_file, filenames):
    for filename in filenames:
        queue_file.put(filename)
    print "done enqueue"

    return queue_file


def test():
    # for i in range(1000000):
    #     print "testing!"
    pass

def test2(str, str2):
    for i in range(10):
        print str + ' ' + str2


def process_q_data(q_data_pcl, q_data_states, q_data_labels):
    global lock
    lock.acquire()
    data_pcl = q_data_pcl.get()
    data_states = q_data_states.get()
    data_labels = q_data_labels.get()
    lock.release()
    print data_pcl[0], data_states[0], data_labels[0]


if __name__ == '__main__':
    # FIFO queues for filenames
    q_file_pcl = Queue.Queue(2)

    # FIFO queues for data
    q_data_pcl = Queue.Queue(2)
    q_data_labels = Queue.Queue(2)
    q_data_states = Queue.Queue(2)

    thread_enq_pcl = MyThread(enqueue_file, args=(q_file_pcl, file_path_list_pcl))
    thread_enq_pcl.start()

    pool = Pool(processes=3)
    pool_get_data = Pool(processes=1)
    # pool.apply_async(test2)
    begin_time = time.time()
    res = []
    for i_pool in range(len(file_path_list_pcl)):
        # pool.apply_async(test)
        filename_pcl, filename_states, filename_labels = \
            file_path_list_pcl.pop(), file_path_list_states.pop(), file_path_list_labels.pop()
        pool.apply_async(read_all_three_csvs, args=(filename_pcl, filename_states, states_num_one_line,
                                                    filename_labels, labels_num_one_line))
    pool_get_data.apply_async(monitor_and_get_data)
    pool_get_data.close()
    pool.close()
    pool.join()
    pool_get_data.join()
    time_spend = time.time() - begin_time
    print "------------------------- time : " + str(time_spend) + "-------------------------"


    print "hhh"
