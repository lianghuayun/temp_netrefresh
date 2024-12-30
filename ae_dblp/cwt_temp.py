import pywt
import numpy as np
import matplotlib.pyplot as plt

from opt import args

from kymatio import Scattering1D
# x = np.arange(512)
# y = np.sin(2*np.pi*x/32)

args.data_path = 'data/{}.txt'.format(args.name)
args.label_path = 'data/{}_label.txt'.format(args.name)
args.model_save_path = 'model/model_save_ae/{}_ae.pkl'.format(args.name)

x = np.loadtxt(args.data_path, dtype=float)
y = np.loadtxt(args.label_path, dtype=int)
y_0 = np.where(y == 0)
y_1 = np.where(y == 1)
y_11 = np.where(y == -1)
y_2 = np.where(y == 2)
y_22 = np.where(y == -2)
wavelet = "cmor1.5-1.0"

log_eps = 1e-6
# scattering = Scattering1D(J=4, shape=x[0].shape)
# x = scattering(x)[:,0,:]

coef, freqs = pywt.cwt(x[0], np.arange(1, 401), wavelet)
coef = np.abs(coef)
plt.matshow(coef)
plt.show()
# j = 0
# for i in range(5):
#     coef, freqs = pywt.cwt(x[y_22[0][j]], np.arange(1, 401), wavelet)
#     j = j+1
#     coef = np.abs(coef)
#     # coef = np.resize(coef, (200, 200))
#     plt.matshow(coef)
# plt.show()
j = 0
for i in range(5):
    coef, freqs = pywt.cwt(x[y_1[0][j]], np.arange(1, 401), wavelet)
    j = j+1
    coef = np.abs(coef)
    plt.matshow(coef)

# j = 0
# for i in range(5):
#     coef, freqs = pywt.cwt(x[y_11[0][j]], np.arange(1, 26), wavelet)
#     j = j+1
#     coef = np.abs(coef)
#     plt.matshow(coef)
# plt.show()
# j = 0
# for i in range(5):
#     coef, freqs = pywt.cwt(x[y_2[0][j]], np.arange(1, 26), wavelet)
#     j = j+1
#     coef = np.abs(coef)
#     plt.matshow(coef)
# plt.show()
j = 0
for i in range(5):
    coef, freqs = pywt.cwt(x[y_22[0][j]], np.arange(1, 401), wavelet)
    j = j+1
    coef = np.abs(coef)
    plt.matshow(coef)
plt.show()
# coef, freqs=pywt.cwt(x[0],np.arange(1, 256), wavelet)
# coef = np.abs(coef)
# plt.matshow(coef)
#
# coef, freqs=pywt.cwt(x[1],np.arange(1, 256), wavelet)
# coef = np.abs(coef)
# plt.matshow(coef)
#
# coef, freqs=pywt.cwt(x[2],np.arange(1, 256), wavelet)
# coef = np.abs(coef)
# plt.matshow(coef)
#
# coef, freqs=pywt.cwt(x[3],np.arange(1, 256), wavelet)
# coef = np.abs(coef)
# plt.matshow(coef)
#
# coef, freqs=pywt.cwt(x[4],np.arange(1, 256), wavelet)
# coef = np.abs(coef)
# plt.matshow(coef)
# plt.show()
# import pywt
# import numpy as np
# import matplotlib.pyplot as plt
# t = np.linspace(-1, 1, 200, endpoint=False)
# sig = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
# plt.figure(1)
# plt.plot(sig)
# plt.figure(2)
# widths = np.arange(1, 31)
# cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')
# plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
#            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
#
# import pywt
#
# def gaussian(x, x0, sigma):
#     return np.exp(-np.power((x - x0) / sigma, 2.0) / 2.0)
#
#
# def make_chirp(t, t0, a):
#     frequency = (a * (t + t0)) ** 2
#     chirp = np.sin(2 * np.pi * frequency * t)
#     return chirp, frequency
#
# # generate signal
# time = np.linspace(0, 1, 2000)
# chirp1, frequency1 = make_chirp(time, 0.2, 9)
# chirp2, frequency2 = make_chirp(time, 0.1, 5)
# chirp = chirp1 + 0.6 * chirp2
# chirp *= gaussian(time, 0.5, 0.2)
#
# # plot signal
# fig, axs = plt.subplots(2, 1, sharex=True)
# axs[0].plot(time, chirp)
# axs[1].plot(time, frequency1)
# axs[1].plot(time, frequency2)
# axs[1].set_yscale("log")
# axs[1].set_xlabel("Time (s)")
# axs[0].set_ylabel("Signal")
# axs[1].set_ylabel("True frequency (Hz)")
# plt.suptitle("Input signal")
# plt.show()
# # perform CWT
# wavelet = "cmor1.5-1.0"
# # logarithmic scale for scales, as suggested by Torrence & Compo:
# widths = np.geomspace(1, 1024, num=100)
# sampling_period = np.diff(time).mean()
# cwtmatr, freqs = pywt.cwt(chirp, widths, wavelet, sampling_period=sampling_period)
# # absolute take absolute value of complex result
# cwtmatr = np.abs(cwtmatr[:-1, :-1])
#
# # plot result using matplotlib's pcolormesh (image with annoted axes)
# fig, axs = plt.subplots(2, 1)
# pcm = axs[0].pcolormesh(time, freqs, cwtmatr)
# axs[0].set_yscale("log")
# axs[0].set_xlabel("Time (s)")
# axs[0].set_ylabel("Frequency (Hz)")
# axs[0].set_title("Continuous Wavelet Transform (Scaleogram)")
# fig.colorbar(pcm, ax=axs[0])
#
# # plot fourier transform for comparison
# from numpy.fft import rfft, rfftfreq
#
# yf = rfft(chirp)
# xf = rfftfreq(len(chirp), sampling_period)
# plt.semilogx(xf, np.abs(yf))
# axs[1].set_xlabel("Frequency (Hz)")
# axs[1].set_title("Fourier Transform")
# plt.tight_layout()
# plt.show()
