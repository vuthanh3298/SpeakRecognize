from __future__ import division
from features import mfcc
from operator import add
import scipy.io.wavfile as wav
import numpy as np

(rate,sig) = wav.read("test_files/alo-test.wav")
duration = len(sig)/rate
mfcc_feat = mfcc(sig,rate,winlen=duration/20,winstep=duration/20)
s = mfcc_feat[:20]
st = []
for elem in s:
	st.extend(elem)

st /= np.max(np.abs(st),axis=0)
	
with open("mfccData/alo_test.npy", 'wb') as outfile:
		np.save(outfile,st)
