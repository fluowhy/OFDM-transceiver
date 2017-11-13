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

def s2p(x):
	"""Conversor serial a paralelo."""
	n = int(len(x)/2)
	xr = np.reshape(x,(n,2))
	return xr


def mapping(x, mapa):
	"""Mapa de bits a simbolos QPSK."""
	return np.array([mapa[tuple(xi)] for xi in x])


def fft(x):
	"""Transformada rapida de Fourier."""
	xf = np.fft.fft(x)
	xmax = np.max(np.abs(xf))
	y = np.fft.fftshift(20*np.log10(np.abs(xf)/xmax))
	return y


def prefix(x, n):
	"""Prefijo de la señal OFDM."""
	a1 = x[-1*n:]
	X = np.concatenate((a1, x))
	return X


def raised(x, n, b): # b<0.7
	"""Ventana raised cosine."""
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
	"""Norma de un vector."""
	fun = y - np.dot(x, h)
	return np.real(np.dot(fun, np.conj(fun)))


def LMS(y, x, alpha=1e-3, tol=1e-2):
	"""Estimador de minimos cuadrados."""
	q = tol + 1	
	hn1 = np.zeros(len(y))
	while q>tol:
		hn2 = hn1 + 2*alpha*np.dot((y - np.dot(x,hn1)).T, x)
		q = np.abs(norma(y, x, hn1) - norma(y, x, hn2))
		hn1 = hn2
		#print 'NO CONVERGE'
		#print hn1
	return hn1


def DEC(x,y):
	"""Mecanismo de decision sobre el simbolo recibido en base a la distancia al simbolo."""
	yy = []
	for xi in x:
		yy.append(np.abs(y-xi)**2)
	yy = np.array(yy).T
	args = np.argmin(yy, axis=1)
	return x[args]


def inter(h, p, f):
	"""Interpola (lineal) la respuesta en frecuencia del canal"""
	hnew = []
	Nf = int(f.shape[0]*0.5)
	Np = int(p.shape[0]*0.5)
	f = np.concatenate((f[Nf:], f[:Nf]))
	p = np.concatenate((p[Np:], p[:Np]))
	h = np.concatenate((h[Np:], h[:Np]))
	hnew = np.interp(f, p, h)
	return np.fft.fftshift(f), np.fft.fftshift(hnew)


def demapping(x, mapa):
	"""Realiza demapping."""
	X = []
	for xi in x:
		Xi = []
		for xir, xii in zip(np.real(xi), np.imag(xi)):
			Xi.append(mapa[xir, xii])
		Xi = np.array(Xi)
		X.append(Xi)
	X = np.array(X)
	return X

def par2ser(x):
	"""Conversor paralelo serial."""
	X = []
	for xi in x:
		Xi = []
		for xii in xi:
			Xi.append(xii[0])
			Xi.append(xii[1])
		Xi = np.array(Xi)		
		X.append(Xi)			
	return np.array(X)


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
cons = {(0, 0):  1+1j, 
	(0, 1): -1+1j,
	(1, 1): -1-1j,
	(1, 0):  1-1j}

decons = {( 1,  1): np.array([0, 0]), 
	  (-1,  1): np.array([0, 1]), 
	  (-1, -1): np.array([1, 1]),
	  ( 1, -1): np.array([1, 0])}

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
beta = 0.1 #<= 0.7
# numero de muestras prefijos
ene = 16
# vector de simbolos enviados 
SN = []
#informacion serial
DS = []

# crea la senal OFDM desde la generaciond de los datos hasta windowing
for i in range(k):
	# senal a enviar, aleatoria
	senal = np.random.binomial(1, 0.5, P*u)
	DS.append(senal)
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

DS = np.array(DS)
SN = np.array(SN)
#plt.plot(S, linewidth=0.5)
#plt.show()

#################################
# Senal en pasa banda
#################################

#s_i = np.real(S)
#s_q = np.imag(S)

# periodo de todos los simbolos
T = Tfft/64*len(S)
tc = np.linspace(0, T, len(S))
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

# potencia promedio de la senal
signal_power = np.trapz(S**2, tc)/T
# snr en dB
vec_snr = np.arange(-10, 31, 1)
BER = []
# selecciona una muestra aleatoria sobre la cual comenzar el removal. 
t = int(np.random.uniform()*ene)
t = input('t') #16
print 'FFT time synchronization error:', 16 - t, 'muestras'
# pilotos y nulls
pilotos = np.concatenate((pilotos, [33, 41, 49, 57]))
pilotos = np.sort(pilotos)
pilotos1 = np.array([7, 15, 23, 31, -31, -23, -15, -7])
datos1 = np.delete(np.arange(0, 64, 1), [0, 32])
freq = np.concatenate((np.arange(1, 32, 1), np.arange(-31, 0,1)))
for snr in vec_snr:
	# ruido AWGN
	sigma = 10**(- snr * 0.1) * signal_power
	nt = np.random.normal(loc=0, scale=np.sqrt(sigma), size=len(S))
	# potencia promedio del ruido
	noise_power = np.trapz(nt**2, tc)/1.5
	# snr promedio
	SNR = 10*np.log10(signal_power/noise_power)
	#print snr, 'dB' 
	#print SNR, 'dB'
	# canal
	h = 1
	# rudio + canal
	r = np.convolve(S, h, mode='same') + nt
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
	mu = np.arange(0, len(r), 1)

	# recorre los 100 simbols ofdm
	block_fft = []
	block_fft_raw = []
	
	sim = mapping([[0, 0], [0, 1], [1, 1], [1, 0]], cons)
	plt.clf()
	for i in range(k):
		# selecciona un bloque de largo m=64 partiendo en t
		# arregla error de sincronizacion
		block = r[t+(m + ene)*i:t+(m + ene)*i+m]
		# DFT
		blockfft = np.fft.fft(block, n=m)
		# guarda data en bruto	
		block_fft_raw.append(blockfft)
		# channel estimation
		# LMS estimator
		# simbolos enviados por nulls y pilots
		#X = np.diag(SN[i][pilotos])
		#Y = blockfft[pilotos]
		#H = LMS(Y, X)
		H = blockfft[pilotos]/SN[i, pilotos]		
		# interpola el canal
		fi, Hi = inter(H, pilotos1, freq)
		#H=H/np.sqrt(np.real(np.dot(H, np.conj(H))))
		# decide simbolos
		# arg de los no nulls
		arg = np.delete(np.arange(0, 64, 1), [0, 32])
		blockfft[arg] = blockfft[arg]/Hi	
		block_fft.append(DEC(sim, blockfft))
		
			
	block_fft = np.array(block_fft)
	block_fft_raw = np.array(block_fft_raw)
	"""
	# grafica los simbolos en el espacio complejo
	plt.clf()
	plt.scatter(np.real(block_fft_raw)[:,datos1], np.imag(block_fft_raw)[:,datos1], marker='+', linewidth=0.5)
	plt.scatter([1, 1, -1, -1], [1, -1, 1, -1], color='black', marker='x')
	plt.xlabel('real')
	plt.ylabel('imaginaria')
	plt.title('Mapa de simbolos qpsk SNR = '+str(snr)+r' dB $\beta$ = '+str(beta))
	plt.savefig('/home/mauricio/Documents/Uni/OFDM/Images/symbols_'+str(snr))
	"""
	# determina bits erroneos
	datos_tot = np.delete(np.arange(0, 64, 1), pilotos)
	# selecciona carriers de datos
	rec = block_fft[:, datos]
	# demap
	rec_demap = demapping(rec, decons)
	# conversion paralela-serial
	DSR = par2ser(rec_demap)
	# bit erroneos
	ber_per_symbol = np.sum(DSR!=DS, axis=1)/P/u
	ber_avg = np.mean(ber_per_symbol)
	BER.append(ber_avg)
BER = np.array(BER)
np.savetxt('BER_t_'+str(16-t), BER)
plt.clf()
plt.scatter(vec_snr, BER, marker='x')
plt.xlabel('SNR dB')
plt.ylabel('BER')
plt.title(r'BER $\beta = $'+str(beta))
plt.grid()
plt.show()



"""
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
"""
plot impulso

plt.stem(datos*1e-1, np.ones(len(datos)), 'b', markerfmt='bo', label='datos', basefmt=" ")
"""



