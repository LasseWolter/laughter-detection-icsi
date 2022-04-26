import librosa
import pandas as pd
import time


def read_from_offset(offset, duration, iterations):
    tot_time = 0

    for i in range(0, iterations):
        start = time.time()
        librosa.load("./Btr002_chan3.sph", sr=16000, offset=offset, duration=duration)
        time_taken = time.time() - start
        tot_time += time_taken

    av_time = tot_time / iterations
    print(f"-------LOAD FROM OFFSET {offset}--------")
    print(
        f"Ran {iterations} iterations:"
        f"\nTotal time: {tot_time:.2f}s"
        f"\nAv_time: {av_time:.2f}s"
        f"\nRTF: {av_time/duration:.2f}"
    )


print("SEGMENT LENGTH 1")
read_from_offset(0, 1, 30)
read_from_offset(300, 1, 30)
read_from_offset(700, 1, 30)
read_from_offset(1000, 1, 30)
read_from_offset(2000, 1, 30)
read_from_offset(3000, 1, 30)
read_from_offset(4000, 1, 30)
read_from_offset(5000, 1, 30)


# print("SEGMENT LENGTH 20")
# read_from_offset(0, 20, 30)
# read_from_offset(300, 20, 30)
# read_from_offset(700, 20, 30)
# read_from_offset(1000, 20,30)
# read_from_offset(2000, 20,30)
# read_from_offset(3000, 20,30)
# read_from_offset(4000, 20,30)
# read_from_offset(5000, 20,30)
