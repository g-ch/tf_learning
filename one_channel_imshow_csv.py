import numpy as np
import sys
import csv
import cv2
import math

input_dimension_x = 256
input_dimension_y = 192
input_channel = 1

img_wid = input_dimension_x
img_height = input_dimension_y

file_path_list_images = [
    # "/home/ubuntu/chg_workspace/data/new_map_with_deepth_img/deepth_rgb_semantics/gazebo_rate_092/yhz/short/03/semantics_2019_03_11_22:23:38.csv",
   "/home/ubuntu/chg_workspace/data/new_map_with_deepth_img/deepth_rgb_semantics/gazebo_rate_092/yhz/long_good/02/semantics_2019_03_11_10:29:56.csv"
]


def read_img_one_channel(filename_img, house, seq, image_num_house):
    maxInt = sys.maxsize
    decrement = True

    clouds = open(filename_img, "r")
    img_num = len(clouds.readlines())
    clouds.close()
    data_img = np.zeros([img_num, img_height, img_wid, 1])

    while decrement:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        decrement = False
        try:
            print "begin read img data.."
            csv.field_size_limit(maxInt)

            with open(filename_img, mode='r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                i_row = 0
                for row in csv_reader:
                    for i in range(img_height):
                        for j in range(img_wid):
                            data_img[i_row, i, j, 0] = row[i * img_wid + j]
                    i_row = i_row + 1
                # list_result.append(data)
        except OverflowError:
            maxInt = int(maxInt / 10)
            decrement = True

        house[seq] = data_img
        image_num_house[seq] = img_num


if __name__ == '__main__':

    '''Data reading'''
    print "Reading data..."

    file_num = len(file_path_list_images)
    data_house = [0 for i in range(file_num)]
    image_num_house = [0 for i in range(file_num)]

    for file_seq in range(file_num):
        print "reading file " + str(file_seq)
        read_img_one_channel(file_path_list_images[file_seq], data_house, file_seq, image_num_house)

    show_num = 100
    for file_seq in range(file_num):
        image_seq_interval = math.floor(image_num_house[file_seq] / show_num)
        print "image_seq_interval: ", image_seq_interval

        if image_seq_interval == 0:
            print "too few images!"
            break

        images = data_house[file_seq]

        for i in range(show_num):
            image_seq_to_show = int(i * image_seq_interval)

            img = images[image_seq_to_show, :, :, :]

            cv2.imshow("img", img)
            cv2.waitKey()
