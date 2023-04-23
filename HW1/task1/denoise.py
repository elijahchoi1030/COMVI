from random import randrange
import cv2
import numpy as np
import os

from torch import zero_

def task1_2(src_path, clean_path, dst_path):
    """
    This is main function for task 1.
    It takes 3 arguments,
    'src_path' is path for source image.
    'clean_path' is path for clean image.
    'dst_path' is path for output image, where your result image should be saved.

    You should load image in 'src_path', and then perform task 1-2,
    and then save your result image to 'dst_path'.
    """
    noisy_img = cv2.imread(src_path)
    clean_img = cv2.imread(clean_path)
    result_img = None

    # do noise removal

    min_rms = 300

    result = apply_median_filter(noisy_img, 3)
    rms = calculate_rms(clean_img, result)
    if rms < min_rms:
        rms = min_rms
        result_img = result

    result = apply_median_filter(noisy_img, 5)
    rms = calculate_rms(clean_img, result)
    if rms < min_rms:
        rms = min_rms
        result_img = result

    result = apply_median_filter(noisy_img, 7)
    rms = calculate_rms(clean_img, result)
    if rms < min_rms:
        rms = min_rms
        result_img = result

    result = apply_median_filter(noisy_img, 11)
    rms = calculate_rms(clean_img, result)
    if rms < min_rms:
        rms = min_rms
        result_img = result

    result = apply_bilateral_filter(noisy_img, 3, 1, 1)
    rms = calculate_rms(clean_img, result)
    if rms < min_rms:
        rms = min_rms
        result_img = result

    result = apply_bilateral_filter(noisy_img, 5, 1, 1)
    rms = calculate_rms(clean_img, result)
    if rms < min_rms:
        rms = min_rms
        result_img = result

    result = apply_bilateral_filter(noisy_img, 5, 3, 3)
    rms = calculate_rms(clean_img, result)
    if rms < min_rms:
        rms = min_rms
        result_img = result

    result = apply_bilateral_filter(noisy_img, 7, 3, 3)
    rms = calculate_rms(clean_img, result)
    if rms < min_rms:
        rms = min_rms
        result_img = result

    result = apply_bilateral_filter(noisy_img, 7, 5, 5)
    rms = calculate_rms(clean_img, result)
    if rms < min_rms:
        rms = min_rms
        result_img = result


    print("RMS of the image: ", calculate_rms(clean_img, result_img))

    cv2.imwrite(dst_path, result_img)
    pass


def apply_median_filter(img, kernel_size):
    """
    You should implement median filter using convolution in this function.
    It takes 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is an int value, which determines kernel size of median filter.

    You should return result image.
    """
    H, W, C = img.shape
    K = int((kernel_size-1)/2)
    padded_img = np.zeros((C, 2*K+H, 2*K+W))
    padded_img[:, K:K+H, K:K+W] = img.transpose(2, 0, 1)

    filter = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    filter = filter.reshape(-1, 1)

    for i in range(H):
        for j in range(W):
            window = padded_img[:, i:i+kernel_size, j:j+kernel_size].reshape(C, -1)
            img[i,j,:] = np.dot(window, filter).reshape(-1)

    return img


def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
    """
    You should implement bilateral filter using convolution in this function.
    It takes at least 4 arguments,
    'img' is source image, and you should perform convolution with bilateral filter.
    'kernel_size' is a int value, which determines kernel size of bilateral filter.
    'sigma_s' is a int value, which is a sigma value for G_s(gaussian function for space)
    'sigma_r' is a int value, which is a sigma value for G_r(gaussian function for range)

    You should return result image.
    """
    H, W, C = img.shape
    K = int((kernel_size-1)/2)
    padded_img = np.zeros((C, 2*K+H, 2*K+W))
    padded_img[:, K:K+H, K:K+W] = img.transpose(2, 0, 1)

    d = np.arange(kernel_size) - K
    dx = np.stack([d for i in range(kernel_size)])
    dy = np.hstack([d.reshape(-1,1) for i in range(kernel_size)])
    G_s = np.exp( - (dx**2 + dy**2) / (2*sigma_s**2) )

    for i in range(H):
        for j in range(W):
            window = padded_img[:, i:i+kernel_size, j:j+kernel_size]
            G_r = np.exp( - (window - img[i,j,:].reshape(-1,1,1))**2  / (2*sigma_r**2) )
            G = G_r * G_s
            img[i,j,:] = ( G*window/G.sum(axis=(1,2)).reshape(-1,1,1) ).sum(axis=(1,2))

    return img


def apply_my_filter(img, kernel_size, sigma):
    """
    You should implement additional filter using convolution.
    You can use any filters for this function, except median, bilateral filter.
    You can add more arguments for this function if you need.

    You should return result image.
    """

    H, W, C = img.shape
    K = int((kernel_size-1)/2)
    padded_img = np.zeros((C, 2*K+H, 2*K+W))
    padded_img[:, K:K+H, K:K+W] = img.transpose(2, 0, 1)

    d = np.arange(kernel_size) - K
    dx = np.stack([d for i in range(kernel_size)])
    dy = np.hstack([d.reshape(-1,1) for i in range(kernel_size)])
    G_s = np.exp( - (dx**2 + dy**2) / (2*sigma**2) )

    for i in range(H):
        for j in range(W):
            window = padded_img[:, i:i+kernel_size, j:j+kernel_size]
            img[i,j,:] = ( G_s*window/G_s.sum(axis=(1,2)).reshape(-1,1,1) ).sum(axis=(1,2))

    return img


    return img


def calculate_rms(img1, img2):
    """
    Calculates RMS error between two images. Two images should have same sizes.
    """
    if (img1.shape[0] != img2.shape[0]) or \
            (img1.shape[1] != img2.shape[1]) or \
            (img1.shape[2] != img2.shape[2]):
        raise Exception("img1 and img2 should have same sizes.")

    diff = np.abs(img1 - img2)
    diff = np.abs(img1.astype(dtype=np.int32) - img2.astype(dtype=np.int32))
    return np.sqrt(np.mean(diff ** 2))


if __name__ == '__main__':
    task = ('cat', 'fox', 'snowman')

    src_root = os.path.join(os.getcwd(), 'test_images')
    clean_root = os.path.join(os.getcwd(), 'test_images')
    dst_root = os.path.join(os.getcwd(), 'output_images')
    os.makedirs(dst_root, exist_ok=True)

    for i in range(len(task)):
        task1_2(os.path.join(src_root, task[i]+'_noisy.jpg'), \
            os.path.join(clean_root, task[i]+'_clean.jpg'), os.path.join(dst_root, task[i]+'_result.jpg'))

