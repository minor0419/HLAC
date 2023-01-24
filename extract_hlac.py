import numpy as np
from scipy import signal
from scipy.sparse import csr_matrix


def extract_hlac(image):
    hlac_filters = [np.array([[False, False, False], [False, True, False], [False, False, False]]),
                    np.array([[False, False, False], [False, True, True], [False, False, False]]),
                    np.array([[False, False, True], [False, True, False], [False, False, False]]),
                    np.array([[False, True, False], [False, True, False], [False, False, False]]),
                    np.array([[True, False, False], [False, True, False], [False, False, False]]),
                    np.array([[False, False, False], [True, True, True], [False, False, False]]),
                    np.array([[False, False, True], [False, True, False], [True, False, False]]),
                    np.array([[False, True, False], [False, True, False], [False, True, False]]),
                    np.array([[True, False, False], [False, True, False], [False, False, True]]),
                    np.array([[False, False, True], [True, True, False], [False, False, False]]),
                    np.array([[False, True, False], [False, True, False], [True, False, False]]),
                    np.array([[True, False, False], [False, True, False], [False, True, False]]),
                    np.array([[False, False, False], [True, True, False], [False, False, True]]),
                    np.array([[False, False, False], [False, True, True], [True, False, False]]),
                    np.array([[False, False, True], [False, True, False], [False, True, False]]),
                    np.array([[False, True, False], [False, True, False], [False, False, True]]),
                    np.array([[True, False, False], [False, True, True], [False, False, False]]),
                    np.array([[False, True, False], [True, True, False], [False, False, False]]),
                    np.array([[True, False, False], [False, True, False], [True, False, False]]),
                    np.array([[False, False, False], [True, True, False], [False, True, False]]),
                    np.array([[False, False, False], [False, True, False], [True, False, True]]),
                    np.array([[False, False, False], [False, True, True], [False, True, False]]),
                    np.array([[False, False, True], [False, True, False], [False, False, True]]),
                    np.array([[False, True, False], [False, True, True], [False, False, False]]),
                    np.array([[True, False, True], [False, True, False], [False, False, False]])]

    masks = ['000010000', '000011000', '001010000', '010010000', '100010000',
             '000111000', '001010100', '010010010', '100010001', '001110000',
             '010010100', '100010010', '000110001', '000011100', '001010010',
             '010010001', '100011000', '010110000', '100010100', '000110010',
             '000010101', '000011010', '001010001', '010011000', '101010000']


    result = []
    image = np.uint8(image)
    hlac_filters = np.uint8(hlac_filters)
    '''
    j = 0
    for mask in masks:
        i = 0
        for i in range(3):
            k = 0
            for k in range(3):
                if mask[i *3 + k] == '0':
                    hlac_filters[j][i][k] = False
                else:
                    hlac_filters[j][i][k] = True
                k += 1
            i += 1
        j += 1
    '''
    for filter in hlac_filters:
        feature_map = signal.convolve2d(image, filter, mode='valid')
        A = csr_matrix(feature_map)
        # マスクと一致する数を集計
        count = A.count_nonzero()
        result.append(count)
    return result
