from re import L
import cv2
import numpy as np

## Do not erase or modify any lines already written
## Each noise function should return image with noise

def add_gaussian_noise(image):
    # Use mean of 0, and standard deviation of image itself to generate gaussian noise

    H, W = image.shape
    noise = np.random.normal(0, image.std(), H*W).reshape(H, W)
    image = image + noise
    
    return image

def add_uniform_noise(image):
    # Generate noise of uniform distribution in range [0, standard deviation of image)

    H, W = image.shape
    noise = np.random.uniform(0, image.std(), H*W).reshape(H, W)
    image = image + noise
    
    return image
 
def apply_impulse_noise(image):
    # Implement pepper noise so that 20% of the image is noisy

    H, W = image.shape

    image = image * np.random.uniform(0.25, 1.5, H*W).round().reshape(H, W)

    '''# 0 -> pixel value 255
    # 0.3 -> pixel value 0
    # 0.6 -> not changing the value
    zeros = np.random.choice([0, 0.3, 0.6], H*W, p=(0.1, 0.1, 0.8))
    image = image * zeros.round().reshape(H, W)                         # 20% goes 0 
    highs = (( np.ones(H*W) - (zeros*2).round() ) * 255).reshape(H,W)   # 0->255, 0.3->0, 0.6->0
    image = image + highs                                               # 10% goes 255 '''

    return image


def rms(img1, img2):
    # This function calculates RMS error between two grayscale images. 
    # Two images should have same sizes.
    if (img1.shape[0] != img2.shape[0]) or (img1.shape[1] != img2.shape[1]):
        raise Exception("img1 and img2 should have the same sizes.")

    diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32))

    return np.sqrt(np.mean(diff ** 2))


if __name__ == '__main__':
    np.random.seed(0)
    original = cv2.imread('bird.jpg', cv2.IMREAD_GRAYSCALE)
    
    gaussian = add_gaussian_noise(original.copy())
    print("RMS for Gaussian noise:", rms(original, gaussian))
    cv2.imwrite('gaussian.jpg', gaussian)
    
    uniform = add_uniform_noise(original.copy())
    print("RMS for Uniform noise:", rms(original, uniform))
    cv2.imwrite('uniform.jpg', uniform)
    
    impulse = apply_impulse_noise(original.copy())
    print("RMS for Impulse noise:", rms(original, impulse))
    cv2.imwrite('impulse.jpg', impulse)