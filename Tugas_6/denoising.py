import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy import fftpack
from matplotlib.colors import LogNorm


image = imageio.imread("man.jpeg", as_gray=True).astype("float64")/255
plt.figure()
plt.imshow(image, plt.cm.gray)

image_fft = fftpack.fft2(image)

def plot_spectrum(image_fft):    
    plt.figure(figsize=(10,10))    
    plt.imshow(np.abs(image_fft), norm=LogNorm(vmin=5),cmap=plt.cm.afmhot), 
    plt.colorbar()

# Menunjukkan frekuensi spektrum dari gambar di atas
im_fft = fftpack.fft2(image)

plt.figure()
plot_spectrum(fftpack.fftshift(image_fft))
plt.title('Spectrum with Fourier transform')

# Mendefinisikan koefisien pecahan
keep_fraction = 0.1
im_fft2 = im_fft.copy()

# Atur Row dan Column ke jumlah baris dan kolom dari array. 
r, c = im_fft2.shape

# Atur semua Row dan Column menjadi nol dengan indeks antara r*keep_fraction and r*(1-keep_fraction)
im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0

plt.figure()
plot_spectrum(fftpack.fftshift(im_fft2))
plt.title('Filtered Spectrum')

# Rekonstruksti Citra
im_new = fftpack.ifft2(im_fft2).real

# Menampilkan perbandingan citra asli dan citra yang telah direkonstruksi
fig, ax = plt.subplots(1,2, figsize=(10,10))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(im_new, cmap='gray')
ax[1].set_title('Reconstructed Image')

# Mengekstrak file gambar
plt.savefig('comparison.png', dpi=300, bbox_inches='tight')