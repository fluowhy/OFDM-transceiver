from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy import signal


"""FALTA"""
"""
-Agregar ventana rised cosine del profe
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
	X = np.concatenate((np.concatenate((a2, x)), a1))
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


# fuente http://dspillustrations.com/pages/posts/misc/python-ofdm-example.html

df = 100 # distancia entre portadoras
fc = 7e3 # portadora
Tfft = 1./df # tiempo de envio de constelacion
Twin = 0 #Tfft/5. # tiempo de la ventana
Tg = 0 # Tfft/4. # tiempo de guarda
T = Twin + Tg + Tfft # tiempo del simbolo ofdm

fs = 44100 # frecuencia de mustreo
ts = 1./fs # periodo de muestreo

n_win = int(np.around(Twin/ts,0))
n_g = int(np.around(Tg/ts,0))
n_fft = int(np.around(Tfft/ts, 0))

cons = {(0,0): 1+1j, # constelacion qpsk
	(0,1): -1+1j,
	(1,1): -1-1j,
	(1,0): 1-1j}

m = 64 # portadoras, minimo 16
P = int(m*7/16 - 1) # portadoras de datos
u = 2 # 2 bits por simbolo qpsk
dt = Tfft/m
pc = 16
wi = 16

carriers = np.arange(-m/2, m/2, 1)*df # frecuencia carriers
pilotos = np.arange(7, int(m/2), 8) # indices pilotos
datos = np.arange(0,int(m/2), 1) # indices datos
datos = np.delete(datos, pilotos)
datos = np.delete(datos, 0)

hann = signal.hann(m) # window con m muestras
nuttall = signal.nuttall(m)

nombre = 'hann'
if nombre=='hann':
	window = hann
elif nombre=='nuttall':
	window = nuttall
elif nombre=='rect':
	window = 1

S = []
k = 100 # ofdm simbolos
NT = m + pc + 2*wi

beta = 0.5 # ventana raised cosine
ene = 16 # numero de muestras prefijos

for i in range(k):
	senal = np.random.binomial(1, 0.5, P*u) # general vector aleatorio de largo m*u=128
	s_p = s2p(senal) # serial a paralelo
	s_map = mapping(s_p, cons) # mapping	
	#s_map[[-1,-2,-3,-4]]=0
	sn = np.zeros(m)+0j
	sn[datos] = s_map
	sn[pilotos] = cons[0,0] # pilotos envian simbolo 00
	sn[int(m/2)+1:] = np.conj(np.flipud(sn[1:int(m/2)]))
	s_ofdm_t = np.fft.ifft(sn, n=len(sn)) # ifft
	s_ofdm_t_cp = prefix(s_ofdm_t, ene) # prefijo ciclico
	s_ofdm_t_r = raised(np.real(s_ofdm_t), ene, beta) # windowing	
	
	#s_win = win(wi, s_ofdm_t_cp)	
	S = np.concatenate((S, s_ofdm_t_r)) # senal temporal de k simbolos	

#plt.plot(S, linewidth=0.5)
#plt.show()

####################
# DAC
####################

nr = int(np.around(fs/len(S), 0)) 
print nr
s = np.repeat(S, nr) # DAC, repite simbolos de manera que se muestree segun frecuencia del DAC fs
b = signal.firwin(200, 3200, window='hann',nyq=fs/2) # coef filtro
sf = signal.convolve(s, b, mode='same', method='direct') # senal filtrada


plt.clf()
#plt.plot(np.fft.fftshift(np.fft.fftfreq(len(S), d=Tfft*k/len(S))), fft(S), linewidth=0.5)
#plt.show()

#################################
# Senal en pasa banda
#################################

s_i = np.real(sf)
s_q = np.imag(sf)

tc = np.linspace(0, T*k, len(sf))
cc = np.cos(2*np.pi*fc*tc)
cs = np.sin(2*np.pi*fc*tc)

sim = s_i*cc
sqm = s_q*cs

sm = sim

plt.clf()
plt.plot(np.fft.fftshift(np.fft.fftfreq(len(sm), d=Tfft*k/len(sm))), fft(sm), linewidth=0.5)
plt.ylim([-140, 0])
plt.title(r'$\beta=$'+str(beta))
plt.savefig('beta_'+str(int(beta*10)))


# sonido
# fuente https://stackoverflow.com/questions/10357992/how-to-generate-audio-from-a-numpy-array

#data = sm # 44100 samples between -1 and 1
#scaled = np.int16(data/np.max(np.abs(data)) * 32767)
#write('test.wav', 44100, scaled)

############################
# Canal
############################



#######################################################
# Receptor
#######################################################




