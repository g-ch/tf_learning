import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv


states_filename = "/home/ubuntu/chg_workspace/data/csvs/1_1/uav_data_2018_12_01_11:03:07.csv"
labels_filename = "/home/ubuntu/chg_workspace/data/csvs/1_1/label_data_2018_12_01_11:03:07.csv"
states_num_one_line = 11
labels_num_one_line = 4

def read_others(data, filename, num_one_line):
    maxInt = sys.maxsize
    decrement = True
    while decrement:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        decrement = False
        try:
            csv.field_size_limit(maxInt)
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

    return True


if __name__ == '__main__':
    states = open(states_filename, "r")
    states_num = len(states.readlines())
    states.close()
    states_mat = np.zeros([states_num, states_num_one_line])
    read_others(states_mat, states_filename, states_num_one_line)

    labels = open(labels_filename, "r")
    labels_num = len(labels.readlines())
    labels.close()
    labels_mat = np.zeros([labels_num, labels_num_one_line])
    read_others(labels_mat, labels_filename, labels_num_one_line)

    compose_num = [256]

    print states_mat[:, 10]

    # concat for input2
    states_input = np.concatenate([np.reshape(states_mat[:, 10], [states_num, 1]) for i in range(compose_num[0])], axis=1)  # vel_odom

    labels_ref = labels_mat[:, 0:2]  # vel_cmd, angular_cmd

    print states_mat[:, 10]
    # a = [[1,2,3],[4,5,6]]
    # a = np.array(a)
    #
    # b = np.concatenate([a for i in range(5)], axis=1)
    # print b
