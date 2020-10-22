import numpy as np
import cv2
import sys
from tqdm import tqdm

def my_dct(mat):
    # 先将mat统一为3个维度
    if len(mat.shape) == 2:
        mat = np.expand_dims(mat, 2)

    num_rows, num_cols, num_channels = mat.shape
    # 如果不满8就补全为8的倍数
    if (num_rows % 8) != 0 or (num_cols % 8) != 0:
        tmp = np.zeros((np.int(np.ceil(num_rows/8)*8), np.int(np.ceil(num_cols/8)*8), num_channels))
        tmp[0:num_rows, 0:num_cols, :] = mat
        mat = tmp

    dct_mat = np.zeros((np.int(np.ceil(num_rows/8)*8), np.int(np.ceil(num_cols/8)*8), num_channels))
    for channel in range(num_channels):
        # 大循环一次处理一个channel
        # 对这个矩阵分块，i，j表示块的编号
        for i in range(np.int(np.ceil(num_rows/8))):
            for j in range(np.int(np.ceil(num_cols/8))):
                # 调用dct子函数
                dct_mat[i*8:(i+1)*8, j*8:(j+1)*8, channel] = dct(mat[i*8:(i+1)*8, j*8:(j+1)*8, channel])

    return dct_mat[0:num_rows, 0:num_cols, :]


def my_idct(dct_mat):
    num_rows, num_cols, num_channels = dct_mat.shape
    # 如果不满8就补全为8的倍数
    if (num_rows % 8) != 0 or (num_cols % 8) != 0:
        tmp = np.zeros((np.int(np.ceil(num_rows / 8) * 8), np.int(np.ceil(num_cols / 8) * 8), num_channels))
        tmp[0:num_rows, 0:num_cols, :] = dct_mat
        dct_mat = tmp

    mat_r = np.zeros((np.int(np.ceil(num_rows / 8) * 8), np.int(np.ceil(num_cols / 8) * 8), num_channels))
    for channel in range(num_channels):
        # 大循环一次处理一个channel
        # 对这个矩阵分块，i，j表示块的编号
        for i in range(np.int(np.ceil(num_rows / 8))):
            for j in range(np.int(np.ceil(num_cols / 8))):
                # 调用idct子函数
                mat_r[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8, channel] = idct(
                    dct_mat[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8, channel])

    mat_r = np.around(mat_r)
    return np.array(mat_r[0:num_rows, 0:num_cols, :], np.int)


def dct(mat: np.ndarray):
    assert len(mat.shape) == 2 and mat.shape[0] == mat.shape[1]
    mat = np.array(mat, np.double)
    N = mat.shape[0]
    A = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            if i == 0:
                c = np.sqrt(1 / N)
            else:
                c = np.sqrt(2 / N)
            A[i, j] = c * np.cos((j + 0.5) * np.pi * i / N)

    mat = A.dot(mat).dot(A.T)

    for i in range(N):
        for j in range(N):
            if i + j > 5:
                mat[i, j] = 0

    return mat


def idct(mat: np.ndarray):
    assert len(mat.shape) == 2 and mat.shape[0] == mat.shape[1]

    N = mat.shape[0]
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == 0:
                c = np.sqrt(1 / N)
            else:
                c = np.sqrt(2 / N)
            A[i, j] = c * np.cos((j + 0.5) * np.pi * i / N)

    mat = (A.T).dot(mat).dot(A)

    for i in range(N):
        for j in range(N):
            if mat[i, j] < 0:
                mat[i, j] = 0

    return mat


def my_dft(mat):
    # 先将mat统一为3个维度
    if len(mat.shape) == 2:
        mat = np.expand_dims(mat, 2)

    num_rows, num_cols, num_channels = mat.shape
    # 使用矩阵乘法
    ux_mat = np.zeros((num_rows, num_rows), np.complex)
    vy_mat = np.zeros((num_cols, num_cols), np.complex)
    for i in range(num_rows):
        for j in range(num_rows):
            ux_mat[i, j] = np.exp(-1j*2*np.pi*i*j/num_rows)

    for i in range(num_cols):
        for j in range(num_cols):
            vy_mat[i, j] = np.exp(-1j*2*np.pi*i*j/num_cols)

    dft_mat = np.zeros(mat.shape, np.complex)
    for channel in range(num_channels):
        dft_mat[:, :, channel] = ux_mat.dot(mat[:, :, channel]).dot(vy_mat)

    return dft_mat


def my_idft(dft_mat):
    # 先将mat统一为3个维度
    if len(dft_mat.shape) == 2:
        dft_mat = np.expand_dims(dft_mat, 2)

    num_rows, num_cols, num_channels = dft_mat.shape
    # 使用矩阵乘法
    ux_mat = np.zeros((num_rows, num_rows), np.complex)
    vy_mat = np.zeros((num_cols, num_cols), np.complex)
    for i in range(num_rows):
        for j in range(num_rows):
            ux_mat[i, j] = np.exp(1j*2*np.pi*i*j/num_rows)

    for i in range(num_cols):
        for j in range(num_cols):
            vy_mat[i, j] = np.exp(1j*2*np.pi*i*j/num_cols)

    mat_r = np.zeros(dft_mat.shape, np.complex)
    for channel in range(num_channels):
        mat_r[:, :, channel] = 1/(num_rows*num_cols) * ux_mat.dot(dft_mat[:, :, channel]).dot(vy_mat)

    return np.array(np.real(mat_r), np.int)


if __name__ == '__main__':
    pass
