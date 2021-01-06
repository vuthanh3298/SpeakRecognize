import numpy as np 

f1 = open("mfccData/orange_mfcc.npy", "rb")
inputArray1  = np.load(f1, allow_pickle=True, encoding="latin1")

print(inputArray1)