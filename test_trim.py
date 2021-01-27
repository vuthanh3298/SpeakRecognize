from pydub import AudioSegment
import scipy.io.wavfile as wav
import numpy as np


def detect_leading_silence(sound, silence_threshold=-28.0, chunk_size=20):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

sound = AudioSegment.from_file("test_files/test2.wav", format="wav")

start_trim = detect_leading_silence(sound)
end_trim = detect_leading_silence(sound.reverse())

duration = len(sound)    
trimmed_sound = sound[start_trim:duration-end_trim]

# print((trimmed_sound.get_array_of_samples()))

samples = np.array(trimmed_sound.get_array_of_samples())

wav.write("test_files/trimed.wav", 8000, samples.astype("int16"))


