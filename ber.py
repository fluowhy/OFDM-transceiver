# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy import signal
"""
# variacion de beta
plt.clf()
vec_snr = np.arange(-10, 31, 1)
BET = []
for i in range(8):
	t = np.loadtxt('BER_beta_'+str((i+1)*0.1))
    	BET.append(t)
BET = np.array(BET)
k = 0
for i in BET:
	name = r'$\beta = $'+str((k+1)*0.1)
	plt.plot(vec_snr, i, label=name)
	k += 1
plt.legend()
plt.xlabel('SNR dB')
plt.ylabel('BER')
plt.title('Variacion BER')
plt.show()
"""
# variacion de t
plt.clf()
vec_snr = np.arange(-10, 31, 1)
BET = []
for i in range(17):
	t = np.loadtxt('BER_t_'+str(i))
    	BET.append(t)
BET = np.array(BET)
k = 0
for i in BET:
	name = 't_sync = '+str(k)
	plt.plot(vec_snr, i, label=name)
	k += 1
plt.legend()
plt.xlabel('SNR dB')
plt.ylabel('BER')
plt.title(r'Variacion BER $\beta = 0.1$')
plt.show()
