
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft,rfft
from random import randint, uniform
from struct import pack
import seaborn
import pyaudio
import csv


rate = 41400


def playSound(sound):
    p = pyaudio.PyAudio()
    stream = p.open(format = pyaudio.paInt16, 
        channels = 1, 
        rate = 41400, 
        output = True)
    stream.write(sound)
    stream.stop_stream()
    stream.close()
    p.terminate()

def genFFT(sound):
    fft_out = [int(i) for i in abs(rfft(data).real)]
    return fft_out[:10000]

def getNote(string,fret):
    # one/zero base conversion
    string -= 1
    fret -= 1
    strings = [('G',4),('C',4),('E',4),('A',4)]
    notes = ['C','Cs','D','Ds','E','F','Fs','G','Gs','A','As','B']
    
    temp = notes.index(strings[string][0]) + fret
    note = notes[temp % len(notes)]
    octave = (temp // len(notes)) + strings[string][1]
    
    return note + str(octave)

def genHeaders(full):
    headers = ['C4','Cs4','D4','Ds4','E4','F4','Fs4','G4','Gs4','A4','As4','B4','C5','Cs5','D5','Ds5','E5','F5','Fs5','G5','Gs5','A5','As5','B5','C6','Cs6']
    if full:
        for i in range(22050):
            headers.append('fft_' + str(i))
    return headers

def genY(ys):
    headers = genHeaders(False)
    output = [0] * len(headers)
    for y in ys:
        output[headers.index(y)] = 1
    return output

def genRow():
    strings = randint(1,4)
    sound = []
    label = []
    for string in range(1,strings+1):
        fret = randint(1,17)
        rate,data = wav.read('ukulele-training-data/%i-%i.wav' % (string,fret))
        duration = len(data) / rate
        wiggle_room = duration - 1
        delta = uniform(0,wiggle_room)
        start,end = int(delta*rate),int((delta+1)*rate)
        if len(sound) == 0:
            output = data[start:end]
        else:
            output = []
            for one,two in zip(sound,data[start:end]):
                output.append(np.int16((int(one)+int(two))/2))
        sound = np.array(output)
        label.append(getNote(string,fret))
        
    first = sound[:int(rate/2)]
    second = sound[int(rate/4):3*int(rate/4)]
    third = sound[int(rate/2):]
    
    x = list(first + second + third)
    y = genY(label)
    
    return y + x


with open('train_ukulele.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='\\', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(genHeaders(True))
    for i in range(20000000):
        writer.writerow(genRow())
        if i % 2000000 == 0:
            print(str(i/200000) + '%... ', end='', flush=True)

