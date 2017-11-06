# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy import signal


"""Transceiver OFDM"""
"""
Se agregó correción de sincronía de FFT, falta mejorarla
"""
"""
FALTA:	- Estimacion de canal (pilotos).
	- Proponer canal acústico.
"""

def s2p(x): # serial a paralelo
	n = int(len(x)/2)
	xr = np.reshape(x,(n,2))
	return xr


def mapping(x, mapa): # mapea la info a qpsk
	return np.array([mapa[tuple(xi)] for xi in x])


def win(nw, x): # x: simbolo + cp, nw:muestras de ventana
	nx = x.size
	n1 = np.arange(0, nw, 1) 
	n3 = np.arange(nw + nx, 2*nw + nx, 1)
	w1 = 0.5*(1 - np.cos(np.pi*n1/nw))
	w2 = x
	w3 = 0.5*(1 - np.cos(np.pi*(n3 - 2*nw - nx)/nw))
	w = np.concatenate((w1, w2, w3))
	return w


def plot(x, y):
	plt.plot(x, y)
	plt.show()
	return


def fft(x):
	xf = np.fft.fft(x)
	xmax = np.max(np.abs(xf))
	y = np.fft.fftshift(20*np.log10(np.abs(xf)/xmax))
	return y

# agrega prefijo final y comienzo, x senal, n numero de muestras
def prefix(x, n):
	a1 = x[0:n]
	a2 = x[-1*n:]
	X = np.concatenate((a2, x, a1))
	return X


def raised(x, n, b): # b<0.7
	ns = len(x) - 2*n
	bns = int(np.rint(b*ns))
	N1 = np.arange(0, bns, 1.)
	N3 = np.arange(len(x) - bns, len(x), 1.)
	rs2 = np.ones(len(x) - 2*bns)	
	rs1 = 0.5*(1 + np.cos(np.pi*(1 + N1/bns)))
	rs3 = 0.5*(1 + np.cos(np.pi*(N3 - (len(x) - bns))/bns))
	rs = np.concatenate((rs1, rs2, rs3))	
	return rs*x


def norma(y,x,h):
	fun = y - np.dot(x, h)
	return np.real(np.dot(fun, np.conj(fun)))


def LMS(y, x, alpha=1e-3, tol=1e-2):
	q = tol + 1	
	hn1 = np.zeros(len(y))
	while q>tol:
		hn2 = hn1 + 2*alpha*np.dot((y - np.dot(x,hn1)).T, x)
		q = np.abs(norma(y, x, hn1) - norma(y, x, hn2))
		hn1 = hn2
		print 'NO CONVERGE'
		print hn1
	return hn1

def DEC(x,y):
	yy = []
	for xi in x:
		yy.append(np.abs(y-xi)**2)
	yy = np.array(yy).T
	args = np.argmin(yy, axis=1)
	return x[args]


# distancia entre portadoras
df = 100 
# frecuencia pasa banda
fc = 7e3 
# tiempo de cada simbolo OFDM (en DFT)
Tfft = 1./df
# frecuencia y periodo de muestreo sound card (aun no se usa)
fs = 44100
ts = 1./fs # periodo de muestreo

# constelacion QPSK
cons = {(0,0): 1+1j, 
	(0,1): -1+1j,
	(1,1): -1-1j,
	(1,0): 1-1j}

# numero de carriers, minimo 16
m = 64 
# numero de carriers de datos
P = int(m*7/16 - 1)
# 2 bits por simbolo QPSK
u = 2
# espaciado temporal de cada muestra
dt = Tfft/m

# frecuencias dadas a carriers
carriers = np.arange(-m/2, m/2, 1)*df
# indices de las frecuencias piloto
pilotos = np.arange(7, int(m/2), 8)
# indice de frecuencia de datos
datos = np.arange(0,int(m/2), 1)
datos = np.delete(datos, pilotos)
datos = np.delete(datos, 0)

# ventanas para windowing (ya no esta en uso)
"""
hann = signal.hann(m) # window con m muestras
nuttall = signal.nuttall(m)

nombre = 'hann'
if nombre=='hann':
	window = hann
elif nombre=='nuttall':
	window = nuttall
elif nombre=='rect':
	window = 1
"""

# crea vector temporal OFDM
S = []
# numero de simbolos OFDM a enviar
k = 100
# parametro raised cosine
beta = 0.4
# numero de muestras prefijos
ene = 16
# vector de simbolos enviados 
SN = []

# crea la senal OFDM desde la generaciond de los datos hasta windowing
for i in range(k):
	# senal a enviar, aleatoria
	senal = np.random.binomial(1, 0.5, P*u)
	# serial a paralelo		
	s_p = s2p(senal) 
	# mapping
	s_map = mapping(s_p, cons)
	# crea vector de IDFT	
	sn = np.zeros(m)+0j
	# pone en carriers de datos los simbolos qpsk
	sn[datos] = s_map
	# pone en pilotos el simbolo 00 (1+1j)
	sn[pilotos] = cons[0,0]
	# orden para la entrada de la IDFT
	sn[int(m/2)+1:] = np.conj(np.flipud(sn[1:int(m/2)]))
	# IDFT
	s_ofdm_t = np.fft.ifft(sn, n=len(sn))
	# prefijo ciclico, se agrega antes y despues
	s_ofdm_t_cp = prefix(s_ofdm_t, ene)
	# se aplica raised cosine
	s_ofdm_t_r = raised(np.real(s_ofdm_t_cp), ene, beta)
	# concatena cada simbolo y crea un vector de informacion	
	S = np.concatenate((S, s_ofdm_t_r))
	SN.append(sn)

SN = np.array(SN)
#plt.plot(S, linewidth=0.5)
#plt.show()

#################################
# Senal en pasa banda
#################################

#s_i = np.real(S)
#s_q = np.imag(S)

tc = np.linspace(0, Tfft/64*9600, 9600)
"""
cc = np.cos(2*np.pi*fc*tc)
cs = np.sin(2*np.pi*fc*tc)

sim = s_i*cc
sqm = s_q*cs

sm = sim

plt.clf()
plt.plot(S, linewidth=0.5)
plt.plot(cc, linewidth=0.5)
plt.plot(sim, linewidth=0.5)
plt.show()

plt.clf()
plt.plot(np.fft.fftshift(np.fft.fftfreq(len(cc), d=1./1000)), fft(cc), linewidth=0.5)
plt.ylim([-140, 0])
plt.title(r'$\beta=$'+str(beta))
plt.show()
"""

# sonido
# fuente https://stackoverflow.com/questions/10357992/how-to-generate-audio-from-a-numpy-array

#data = sm # 44100 samples between -1 and 1
#scaled = np.int16(data/np.max(np.abs(data)) * 32767)
#write('test.wav', 44100, scaled)

####################
# DAC
####################
"""
nr = int(np.around(fs/len(S), 0)) 
print nr
s = np.repeat(S, nr) # DAC, repite simbolos de manera que se muestree segun frecuencia del DAC fs
b = signal.firwin(200, 3200, window='hann',nyq=fs/2) # coef filtro
sf = signal.convolve(s, b, mode='same', method='direct') # senal filtrada


plt.clf()
#plt.plot(np.fft.fftshift(np.fft.fftfreq(len(S), d=Tfft*k/len(S))), fft(S), linewidth=0.5)
#plt.show()
"""
############################
# Canal
############################

# ruido AWGN 
nt = np.random.normal(loc=0, scale=1e1, size=len(S))
# potencia promedio del ruido
noise_power = np.trapz(nt**2, tc)/1.5
# potencia promedio de la senal
signal_power = np.trapz(S**2, tc)/1.5
# snr promedio
snr = 10*np.log10(signal_power/noise_power)
print snr, 'dB'
# canal
h = 1
# rudio + canal
r = S*h + nt
"""
# plot espectral ruido
plt.clf()
plt.plot(np.fft.fftshift(np.fft.fftfreq(len(nt), d=dt)), fft(nt), linewidth=0.5)
plt.xlabel('Frecuencia Hz')
plt.ylabel('Potencia dB')
plt.title('Ruido')
plt.show()

# plot espectral senal OFDM
plt.clf()
plt.plot(np.fft.fftshift(np.fft.fftfreq(len(S), d=dt)), fft(S), linewidth=0.5)
plt.xlabel('Frecuencia Hz')
plt.ylabel('Potencia dB')
plt.title(r'Senal')
plt.show()

# plot espectral senal OFDM + canal
plt.clf()
plt.plot(np.fft.fftshift(np.fft.fftfreq(len(r), d=dt)), fft(r), linewidth=0.5)
plt.xlabel('Frecuencia Hz')
plt.ylabel('Potencia dB')
plt.title(r'Senal con ruido, snr '+str(np.around(snr, 0))+' dB')
plt.show()
"""
#######################################################
# Receptor
#######################################################

# guard interval removal

# selecciona una muestra aleatoria sobre la cual comenzar el removal. 
t = int(np.random.uniform()*16)
#t = 15
mu = np.arange(0, len(r), 1)

# pilotos y nulls
pilotos = np.concatenate(([0, 32], pilotos, [33, 41, 49, 57]))
pilotos = np.sort(pilotos)

# recorre los 100 simbols ofdm
block_fft = []
block_fft_raw = []
freq = np.concatenate((np.arange(0, 32, 1), np.arange(-32, 0,1)))
sim = mapping([[0, 0], [0, 1], [1, 1], [1, 0]], cons)
for i in range(k):
	# selecciona un bloque de largo 64 partiendo en t
	# arregla error de sincronizacion 
	block = r[t+96*i:t+96*i+64]
	# DFT
	blockfft = np.fft.fft(block, n=len(block))
	# guarda data en bruto	
	block_fft_raw.append(DEC(sim, blockfft))
	# LMS estimator
	# simbolos enviados por nulls y pilots
	X = np.diag(SN[i][pilotos])
	Y = blockfft[pilotos]
	H = LMS(Y, X)
	H=H/np.sqrt(np.real(np.dot(H, np.conj(H))))
	# decide simbolos
	blockfft[pilotos] = blockfft[pilotos]*np.conj(H)
	
	block_fft.append(DEC(sim, blockfft))#*np.exp(1j*2*np.pi*freq*(16 - t)/64))
	
block_fft = np.array(block_fft)
block_fft_raw = np.array(block_fft_raw)

# determina simbolos erroneos

plt.clf()
plt.plot(np.sum(np.equal(block_fft, SN), axis=1)/64*100, label='canal estimado')
plt.plot(np.sum(np.equal(block_fft_raw, SN), axis=1)/64*100, label='sin estimacion')
plt.title('Error OFDM SNR='+str(int(np.rint(snr)))+' dB')
plt.xlabel('simbolo OFDM')
plt.ylabel('error $\%$')
plt.legend()
plt.show()
"""





plt.clf()
plt.plot(np.real(H))
plt.plot(np.imag(H))
plt.show()

# grafica los simbolos en el espacio complejo
#ui = 31
plt.clf()
for i in block_fft:
	plt.scatter(np.real(i), np.imag(i), marker='.')
plt.scatter([1, 1, -1, -1], [1, -1, 1, -1], color='black', marker='x')
plt.xlabel('real')
plt.ylabel('imaginaria')
plt.title('Mapa de simbolos qpsk')
plt.show()

# grafica los pilotos en el espacio complejo + ceros
plt.clf()
for i in block_fft:
	j = i[pilotos]#*np.conj(H)
	plt.scatter(np.real(j[1]), np.imag(j[0]), marker='x', color='red')
	plt.scatter(np.real(j[2]), np.imag(j[1]), marker='x', color='green')
	plt.scatter(np.real(j[3]), np.imag(j[2]), marker='x', color='blue')
	plt.scatter(np.real(j[4]), np.imag(j[3]), marker='x', color='orange')
	plt.scatter(np.real(j[[0, 5]]), np.imag(j[[0, 5]]), marker='x', color='brown')
plt.scatter([0, 1, 1, -1, -1], [0, 1, -1, 1, -1], color='black')
plt.xlabel('real')
plt.ylabel('imaginaria')
plt.title('Mapa de simbolos qpsk pilotos')
plt.show()

"""




