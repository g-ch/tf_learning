import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl


# draw by axis z direction
def compare_draw_3d_to_2d(data1, data2, min_val, max_val, rows, cols, step):
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
    # colors = ['purple', 'yellow']
    # bounds = [min_val, max_val]
    # cmap = mpl.colors.ListedColormap(colors)
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    f, a = plt.subplots(rows, cols, figsize=(cols, rows))

    if (data1.shape[2] / step) > (rows * cols / 2):
        print "Plots index out of range!"
        return

    for i in range(cols):
        for j in range(rows / 2):
            a[2 * j][i].imshow(data1[:, :, (j * cols + i) * step])
            a[2 * j + 1][i].imshow(data2[:, :, (j * cols + i) * step])

    plt.show() #cmap=cmap, norm=norm


def draw_plots(x, y):
    """
    Draw multiple plots
    :param x: should be 2d array
    :param y: should be 2d array
    :return:
    """
    for i in range(x.shape[0]):
        plt.plot(x[i], y[i])

    plt.title("matplotlib")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    data_mat = np.ones([5, 64, 64, 64, 1]) / 2
    data_mat[2, 8, :, :, 0] = 0.0

    decode_pcl = np.zeros([5, 64, 64, 64, 1])
    decode_pcl[2, 8, :, :, 0] = 1.0


    compare_draw_3d_to_2d(data_mat[2,:,:,:,0], decode_pcl[2,:,:,:,0], 0, 1, 4, 8, 4)

    # x = np.array([np.arange(1, 10), np.arange(11, 15), np.arange(15, 20)])
    # y = np.multiply(0.02, x)
    # draw_plots(x, y)
