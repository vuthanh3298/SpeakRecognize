import numpy as np
from scipy.io.wavfile import write

f = open("mfccData/alo_test.npy", "rb")
data = np.load(f, allow_pickle=True, encoding="bytes")

write('alo-test-convert.wav', 44100, data)