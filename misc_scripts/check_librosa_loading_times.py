import librosa
import pandas as pd
import time


def read_from_offset(offset, duration, iterations):
    tot_time=0

    for i in range(0,iterations):
        start = time.time()
        librosa.load('chan1.sph', sr=16000, offset=offset, duration=duration)
        time_taken = time.time() - start
        tot_time += time_taken 

    av_time = tot_time/iterations
    print(f'-------LOAD FROM OFFSET {offset}--------')
    print(f'Ran {iterations} iterations:'
            f'\nTotal time: {tot_time:.2f}s'
            f'\nAv_time: {av_time:.2f}s'
            f'\nRTF: {av_time/duration:.2f}')

    
#print("SEGMENT LENGTH 1")
#read_from_offset(0, 1, 10)
#read_from_offset(390, 1, 10)
#read_from_offset(3360, 1,10)


print("SEGMENT LENGTH 20")
read_from_offset(0, 20, 10)
read_from_offset(390, 20, 10)
read_from_offset(3360, 20,10)
