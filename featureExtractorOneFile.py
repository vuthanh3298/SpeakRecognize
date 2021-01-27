from __future__ import division
from features import mfcc
from operator import add
import scipy.io.wavfile as wav
import numpy as np

(rate,sig) = wav.read("training_sets_respeaker/th-tatdennhabep.wav", 'rb')
duration = len(sig)/rate
mfcc_feat = mfcc(sig,rate,winlen=duration/20,winstep=duration/20)

print(rate)
print(mfcc_feat.shape)

# s = mfcc_feat[:20]

# # print("s \n", s)

# st = []
# for elem in s:
# 	st.extend(elem)

# st /= np.max(np.abs(st),axis=0)
# # data.append(st)
# print(st)
# print(st.shape)
# print(rate)


# print(mfcc_feat.shape)
# print(mfcc_feat)




