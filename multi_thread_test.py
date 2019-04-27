import threading
import numpy as np
import tensorflow as tf
import os

aa = 1

def func(coord, t_id):
    count = 0
    names = globals()
    print(names['n' + str(t_id)])

    while not coord.should_stop():
        # print('thread ID:',t_id, 'count =', count)
        count += 1

        if(count % 10000 == 0):
            global aa
            aa += 1
            # print('thread ID:'+ str(t_id) + 'aa =' +str(aa))

        if(count == 1000000):
            coord.request_stop()


def get_file_size(filePath):
    fsize = os.path.getsize(filePath)
    fsize = fsize/float(1024*1024)
    return round(fsize, 2)


if __name__ == '__main__':
    print get_file_size("/home/ubuntu/chg_workspace/data/new_map_with_deepth_noi_rotate/long_good/01/dep_noi_data_2019_04_23_07:51:16.csv")


    names = globals()
    for i in range(8):
        names['n' + str(i)] = 1994 * i

    coord = tf.train.Coordinator()
    threads = [threading.Thread(target=func, args=(coord, i)) for i in range(8)]

    for t in threads:
        t.start()
    coord.join(threads)
