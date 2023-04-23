import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

##### To-do #####

def fftshift(img):
    '''
    This function should shift the spectrum image to the center.
    You should not use any kind of built in shift function. Please implement your own.
    '''

    F_img = np.fft.fft2(img)
    H, W = img.shape
    h, w = math.floor(H/2), math.floor(W/2)

    img = np.zeros(img.shape, dtype=complex)
    img[0:h, 0:w] = F_img[H-h:H, W-w:W]
    img[h:H, 0:w] = F_img[0:H-h, W-w:W]
    img[0:h, w:W] = F_img[H-h:H, 0:W-w]
    img[h:H, w:W] = F_img[0:H-h, 0:W-w]

    return img

def ifftshift(img):
    '''
    This function should do the reverse of what fftshift function does.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    f_img = np.zeros(img.shape, dtype=complex)
    H, W = img.shape
    h, w = math.floor(H/2), math.floor(W/2)

    f_img[0:H-h, 0:W-w] = img[h:H, w:W]
    f_img[H-h:H, 0:W-w] = img[0:h, w:W]
    f_img[0:H-h, W-w:W] = img[h:H, 0:w]
    f_img[H-h:H, W-w:W] = img[0:h, 0:w]

    img = np.abs(np.fft.ifft2(f_img))

    return img

def fm_spectrum(img):
    '''
    This function should get the frequency magnitude spectrum of the input image.
    Make sure that the spectrum image is shifted to the center using the implemented fftshift function.
    You may have to multiply the resultant spectrum by a certain magnitude in order to display it correctly.
    '''

    F_img = fftshift(img)
    img = 20*np.log10(np.abs(F_img) + 1e-3)

    return img

def low_pass_filter(img, r=30):
    '''
    This function should return an image that goes through low-pass filter.
    '''

    H, W = img.shape
    h, w = math.floor(H/2), math.floor(W/2)

    dw = np.arange(W) - w
    dw = np.stack([dw for i in range(H)])
    dh = (np.arange(H) - h).reshape(-1,1)
    dh = np.hstack([dh for i in range(W)])
    filter = np.array( (dw**2 + dh**2) < r**2, dtype=np.uint8)

    F_img = fftshift(img)
    F_img = F_img * filter
    img = ifftshift(F_img)

    return img

def high_pass_filter(img, r=20):
    '''
    This function should return an image that goes through high-pass filter.
    '''

    H, W = img.shape
    h, w = math.floor(H/2), math.floor(W/2)

    dw = np.arange(W) - w
    dw = np.stack([dw for i in range(H)])
    dh = (np.arange(H) - h).reshape(-1,1)
    dh = np.hstack([dh for i in range(W)])
    filter = np.array( (dw**2 + dh**2) > r**2, dtype=np.uint8)

    F_img = fftshift(img)
    F_img = F_img * filter
    img = ifftshift(F_img)

    return img

def denoise1(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''

    F_img = fftshift(img)
    key = [172, 200, 308, 336]
    keys = [(key[0], key[0]), (key[1], key[1]), (key[0], key[3]), (key[1], key[2]), \
        (key[3], key[0]), (key[2], key[1]), (key[2], key[2]), (key[3], key[3]), ]

    for kH, kW in keys:
        for i in range(5):
            val = np.linspace(F_img[kH+i, kW-1], F_img[kH+i, kW+5], 7)
            F_img[kH+i, kW:kW+5] = val[1:6]

    img = ifftshift(F_img)

    return img

def denoise2(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''

    H, W = img.shape
    h, w = math.floor(H/2), math.floor(W/2)

    dw = np.arange(W) - w
    dw = np.stack([dw for i in range(H)])
    dh = (np.arange(H) - h).reshape(-1,1)
    dh = np.hstack([dh for i in range(W)])
    mask1 = np.floor(np.sqrt(dw**2 + dh**2)) == 27
    mask1 = np.array(mask1, dtype=np.uint8) * 0.75
    mask2 = np.logical_and(np.floor(np.sqrt(dw**2 + dh**2)) > 27, np.logical_or(np.abs(dh) < 20, np.abs(dw) < 15))
    mask2 = np.array(mask2, dtype=np.uint8) * 0.3
    mask = 1 - mask1 - mask2

    cv2.imwrite('mask.png', mask*255)

    F_img = fftshift(img)
    F_img = F_img * mask
    img = ifftshift(F_img)


    return img

#################

# Extra Credit
def dft2(img):
    '''
    Extra Credit. 
    Implement 2D Discrete Fourier Transform.
    Naive implementation runs in O(N^4).
    '''

    H, W = img.shape

    dft_img = np.zeros(img.shape, dtype=np.complex128)
    for h_ in range(H):
        for w_ in range(W):
            for h in range(H):
                for w in range(W):
                    real = img[h, w] * np.cos( - (h_*h/H + w_*w/W) * 2 * np.pi )
                    imag = img[h, w] * np.sin( - (h_*h/H + w_*w/W) * 2 * np.pi )

                    dft_img[h_, w_] += real + imag*1j

    return dft_img

def idft2(img):
    '''
    Extra Credit. 
    Implement 2D Inverse Discrete Fourier Transform.
    Naive implementation runs in O(N^4). 
    '''

    H, W = img.shape

    idft_img = np.zeros(img.shape, dtype=np.complex128)
    for h_ in range(H):
        for w_ in range(W):
            for h in range(H):
                for w in range(W):
                    real = img[h, w] * np.cos( (h_*h/H + w_*w/W) * 2 * np.pi )
                    imag = img[h, w] * np.sin( (h_*h/H + w_*w/W) * 2 * np.pi )

                    idft_img[h_, w_] += real + imag*1j

    return idft_img

def fft2(img):
    '''
    Extra Credit. 
    Implement 2D Fast Fourier Transform.
    Correct implementation runs in O(N^2*log(N)).
    '''
    return img

def ifft2(img):
    '''
    Extra Credit. 
    Implement 2D Inverse Fast Fourier Transform.
    Correct implementation runs in O(N^2*log(N)).
    '''
    return img

if __name__ == '__main__':
    img = cv2.imread('task2_filtering.png', cv2.IMREAD_GRAYSCALE)
    noised1 = cv2.imread('task2_noised1.png', cv2.IMREAD_GRAYSCALE)
    noised2 = cv2.imread('task2_noised2.png', cv2.IMREAD_GRAYSCALE)

    low_passed = low_pass_filter(img)
    high_passed = high_pass_filter(img)
    denoised1 = denoise1(noised1)
    denoised2 = denoise2(noised2)

    # save the filtered/denoised images
    cv2.imwrite('low_passed.png', low_passed)
    cv2.imwrite('high_passed.png', high_passed)
    cv2.imwrite('denoised1.png', denoised1)
    cv2.imwrite('denoised2.png', denoised2)

    # draw the filtered/denoised images
    def drawFigure(loc, img, label):
        plt.subplot(*loc), plt.imshow(img, cmap='gray')
        plt.title(label), plt.xticks([]), plt.yticks([])

    drawFigure((2,7,1), img, 'Original')
    drawFigure((2,7,2), low_passed, 'Low-pass')
    drawFigure((2,7,3), high_passed, 'High-pass')
    drawFigure((2,7,4), noised1, 'Noised')
    drawFigure((2,7,5), denoised1, 'Denoised')
    drawFigure((2,7,6), noised2, 'Noised')
    drawFigure((2,7,7), denoised2, 'Denoised')

    drawFigure((2,7,8), fm_spectrum(img), 'Spectrum')
    drawFigure((2,7,9), fm_spectrum(low_passed), 'Spectrum')
    drawFigure((2,7,10), fm_spectrum(high_passed), 'Spectrum')
    drawFigure((2,7,11), fm_spectrum(noised1), 'Spectrum')
    drawFigure((2,7,12), fm_spectrum(denoised1), 'Spectrum')
    drawFigure((2,7,13), fm_spectrum(noised2), 'Spectrum')
    drawFigure((2,7,14), fm_spectrum(denoised2), 'Spectrum')

    plt.show()