from threading import Thread
from record import record_to_file
from features import mfcc
from anntester_single import *
import scipy.io.wavfile as wav
import playsound as plsnd
import requests as req


if __name__ == '__main__':
    testNet = testInit()

    num_loop = 0
    filename="test_files/test.wav"

    while True:
        # Record to file
        num_loop += 1
        print("please speak a word into the microphone", num_loop)
        record_to_file(filename)

        # Feed into ANN

        filename = "training_sets/alo-1.wav"
        
        inputArray = extractFeature(filename)
        res = feedToNetwork(inputArray,testNet)
        
        print("Detected: Alo")

        outStr = None

        if(res == 0):
            # ban can giup gi?
           
            plsnd.playsound("speak_out_files/bancangiupgi.wav")
            print("Ban can giup gi? ...")

            

            record_to_file(filename)

            if(num_loop == 1) :
                filename = "training_sets/batden-2.wav"
            elif(num_loop == 2):
                filename = "training_sets/tatden-2.wav"
            if(num_loop == 3) :
                filename = "training_sets/batden-2.wav"
            elif(num_loop == 4):
                filename = "training_sets/tatden-2.wav"    
            

            inputArray = extractFeature(filename)
            res = feedToNetwork(inputArray,testNet)
            if res==1:
                outStr  = "Detected: Bat den ";

                print("Detected: Bat den")

                plsnd.playsound("speak_out_files/dabatden.wav")

                req.get("http://192.168.137.1:3000/light?data=1")


            elif res==2:
                outStr  = "Detected: Bat quat";
                plsnd.playsound("speak_out_files/dabatquat.wav")

            elif res==3:
                outStr  = "Detected: Tat den"

                print("Detected: Tat den")
                
                plsnd.playsound("speak_out_files/datatden.wav")

            elif res==4:
                outStr  = "Detected: Tat quat";
                plsnd.playsound("speak_out_files/datatquat.wav")


        print(outStr)