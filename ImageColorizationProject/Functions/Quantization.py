import numpy as np
import cv2
import os

def quantize_ab_channels(ab_channels, bin_width=10):

    grid_size = int(255/bin_width)
    canvas = np.zeros((grid_size, grid_size))
    for i in range(ab_channels.shape[0]):
        for j in range(ab_channels.shape[1]):
            a_bin = int(np.floor(ab_channels[i, j, 0] / bin_width))
            b_bin = int(np.floor(ab_channels[i, j, 1] / bin_width))
            canvas[b_bin, a_bin] += 1
    return np.array(canvas, dtype = np.uint32)


def buildcanvas(data_dir, save = False):

    canvas = np.zeros((25, 25))
    for i, filename in enumerate(os.listdir(data_dir)):
        if filename.endswith('.jpg'):
            filepath = os.path.join(data_dir, filename)
            img = cv2.imread(filepath)[:,:,::-1]
            image = cv2.resize(img, (224,224))
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            L, A, B = cv2.split(lab_image)
            arr1 = np.transpose([A, B], (1, 2, 0))
            canvas += quantize_ab_channels(arr1)

    if save:
        np.save('non_normalized_canvas.npy', canvas)

    return canvas, canvas/np.sum(canvas)

def create_weights(p_tilde, gamma = 0.5, save = False):
    w = np.zeros_like(p_tilde)
    Q = np.count_nonzero(p_tilde)
    for x in range(len(p_tilde)):
        for y in range(len(p_tilde[0])):
            w[x, y] = ((1 - gamma)*p_tilde[x, y] + gamma/Q)**(-1)

    w_tilde = (w * p_tilde)/np.sum(w * p_tilde)
    w_final = w_tilde / p_tilde

    if save :
        np.save('weight_matrix.npy', w)
    
    return w_final


def build_index(canvas):
    index_map = {}
    counter = 0
    for i in range(len(canvas)):
        for j in range(len(canvas[0])):
            if canvas[i, j] != 0:
                index_map[counter] = (i, j)
                counter += 1
    inverse_index_map = {v: k for k, v in index_map.items()}

    return index_map, inverse_index_map


def flatten_weights(inverse_index_map, w):

    w_vector = np.zeros((len(inverse_index_map.keys())))
    for row in range(len(w)):
        for column in range(len(w[0])):
            if (row, column) in inverse_index_map.keys():
                w_vector[inverse_index_map[row, column]] = w[row, column]

    return w_vector
