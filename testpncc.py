
import scipy.io.wavfile as wav
from spafe.utils import vis
from spafe.features.pncc import pncc

    
# read wav 
fs, sig = wav.read("training_sets_respeaker/abcxyz.wav")

sig = sig[:, 0]

print("rate: ", fs)
print("shape sig: ", sig.shape)

duration = len(sig)/fs
print(duration)
winlen=duration/10
winstep=duration/30
print(winlen)
print(winstep)

# compute features
pnccs = pncc(sig=sig, fs=fs, win_len=winlen, win_hop=winstep)

print(pnccs.shape)

# visualize spectogram
# vis.spectogram(sig, fs)
# # visualize features
# vis.visualize_features(pnccs, 'PNCC Index', 'Frame Index')