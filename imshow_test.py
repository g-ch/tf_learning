import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import csv

img_wid = 64
img_height = 24


# draw by axis z direction
def compare_save_3d_to_2d(data1, data2, min_val, max_val, rows, cols, step, name):
    """
    To compare two 3 dimension array by image slices
    :param data1: data to compare, 3 dimension array
    :param data2: should have the same size as data1
    :param min_val: minimum value in data1 and data2
    :param max_val: maximum value in data1 and data2
    :param rows: row number of the figure
    :param cols: col number of the figure
    :param step: step in z axis to show
    :return:
    """
    colors = ['purple', 'yellow']
    bounds = [min_val, max_val]
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    f, a = plt.subplots(rows, cols, figsize=(cols, rows))

    # for scale
    data1_copy = np.array(tuple(data1))
    data2_copy = np.array(tuple(data2))
    data1_copy[0, 0, :] = min_val
    data2_copy[0, 0, :] = min_val
    data1_copy[0, 1, :] = max_val
    data2_copy[0, 1, :] = max_val

    for i in range(cols):
        for j in range(rows / 2):
            a[2 * j][i].imshow(data1_copy[:, :, (j * cols + i) * step])
            a[2 * j + 1][i].imshow(data2_copy[:, :, (j * cols + i) * step])

    #plt.show(cmap=cmap, norm=norm)
    plt.savefig(name)
    plt.cla


def read_pcl(data, filename):
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
                    for i in range(img_wid):
                        for j in range(img_wid):
                            for k in range(img_height):
                                data[i_row, i, j, k, 0] = row[i * img_wid * img_height + j * img_height + k]
                    i_row = i_row + 1
        except OverflowError:
            maxInt = int(maxInt / 10)
            decrement = True

    return data


if __name__ == "__main__":

    # data_mat = np.ones([5, 64, 64, 64, 1]) / 2
    # data_mat[2, 10, :, 2, 0] = 0
    # data_mat[2, 30, :, 4, 0] = 1
    # decode_pcl = np.zeros([5, 64, 64, 64, 1])
    #
    # compare_draw_3d_to_2d(data_mat[2,:,:,:,0], decode_pcl[2,:,:,:,0], 0, 1, 4, 16, 2)
    # print data_mat[2,:,:,:,0]

    file_name = "/home/ubuntu/chg_workspace/data/new_csvs/backward_unable/chg_route1_trial1/pcl_data_2018_12_12_14:03:47.csv"
    clouds = open(file_name, "r")
    img_num = len(clouds.readlines())
    clouds.close()
    data_mat = np.ones([img_num, img_wid, img_wid, img_height, 1])
    data_mat = read_pcl(data_mat, file_name)

    print "data_mat", data_mat

    data_mat[0,5,5,:,0] = 0.14

    path = '/home/ubuntu/chg_workspace/data/plots/input_pcl_test/'
    for i in range(180):
        name = path + str(i) + '.png'
        compare_save_3d_to_2d(data_mat[i*5, :, :, :, 0], data_mat[i*5, :, :, :, 0], 0, 1, 4, 12, 1, name)
