import cv2
import matplotlib.pyplot as plt
import numpy as np


def FFT(img):
    return np.fft.fftshift(np.fft.fft2(img))

def IFFFT(img):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(img)))

def Butterworth_Lowpass(F, D0, n):
    M, N = F.shape
    H = np.zeros((M, N))

    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            H[u, v] = 1/(1+(D/D0)**(2*n))

    G = H * F
    return IFFFT(G)

def Gaussian_Lowpass(F, D0):
    M, N = F.shape
    H = np.zeros((M, N))

    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            H[u, v] = np.exp(-(D**2)/(2*D0**2))
    
    G = H * F
    return IFFFT(G)

img_path = './pattern.tif'
gray_img = cv2.imread(img_path,0)
img = cv2.resize(gray_img,(512,512))

height, width = img.shape

mean = 0
standard_deviation = 25
noise = np.random.randn(height, width)*standard_deviation + mean
noise = noise.astype(np.uint8)
Gnoisy_img = cv2.add(img, noise)

Gnoisy_img_fft = FFT(img)
Gnoisy_img_magnitude = np.log(np.abs(Gnoisy_img_fft)+1)

order = 4
cutoff_frequency = 30
reconstructed_butter_img = Butterworth_Lowpass(Gnoisy_img_fft, cutoff_frequency, order)
reconstructed_gaussian_img = Gaussian_Lowpass(Gnoisy_img_fft, cutoff_frequency)



plt.figure(figsize=(5,5))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

plt.subplot(2,3,1)
plt.title('Original Image')
plt.imshow(img,cmap='gray')

plt.subplot(2,3,2)
plt.title('Gaussian Noisy Image')
plt.imshow(Gnoisy_img,cmap='gray')

plt.subplot(2,3,3)
plt.title('Gaussian Noisy Image FFT')
plt.imshow(Gnoisy_img_magnitude,cmap='gray')

plt.subplot(2,3,4)
plt.title('Reconstructed Butter Image')
plt.imshow(reconstructed_butter_img,cmap='gray')

plt.subplot(2,3,5)
plt.title('Reconstructed Gaussian Image')
plt.imshow(reconstructed_gaussian_img,cmap='gray')

plt.show()