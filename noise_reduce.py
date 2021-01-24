# import noisereduce as nr
import noise_reduce_core as nr
import scipy.io.wavfile as wav
import numpy as np

# load data
rate, data = wav.read("./training_sets/batquat-1.wav")

# data = data.flatten()

print("data: \n")
print(data)

print("\n\nndim: ", data.ndim)


if data.ndim > 1:
	data = data[:, 0]

print("data.shape: ", data.shape)
print("\n\ndata:")
print(data)

# perform noise reduction
# reduced_noise = nr.reduce_noise(audio_clip=reduced_noise.astype('float'), noise_clip=reduced_noise.astype('float'))
reduced_noise = nr.reduce_noise(audio_clip=data.astype('float32'), noise_clip=data.astype('float32'))

print("\n\nreduced_noise:")
print(reduced_noise)

print("\n\nreduced_noise astype('int'):")
print(reduced_noise.astype("int16"))

# reduced_noise = reduced_noise.astype("int16")

wav.write("reduced-noise.wav", rate, reduced_noise.astype(np.int16))

# load data
rate, data = wav.read("reduced-noise.wav")
print("\n\nreduced_noise:")
print(reduced_noise)
