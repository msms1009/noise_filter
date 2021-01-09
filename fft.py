import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy import fftpack
import math
import wave
import struct
import librosa
import pandas as pd
from scipy.io import wavfile
from scipy.signal import lfilter
import noisereduce as nr
from scipy.signal import savgol_filter

def test3():
    samplerate = 44100
    
    spf = wave.open("c.wav", "r")
    signal = spf.readframes(-1)
    sig = np.fromstring(signal, "int32")
    
    n = 100 # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 2
    yy = lfilter(b,a,sig)
    
    wavfile.write("c_.wav", samplerate, yy.astype('int32'))

def test2(): #make high freq
    time_step = 0.01
    samplerate = 44100
    period = 5.

    time_vec = np.arange(0, 20, time_step)

    spf = wave.open("voice.wav", "r")
    signal = spf.readframes(-1)
    sig = np.fromstring(signal, "int32") ################################

    rate, sig_ = wavfile.read("voice.wav")
    
    plt.figure(figsize=(6, 5))
    plt.plot(sig, label='Original signal')

    sig_fft = fftpack.fft(sig)
    
    plt.figure(figsize=(6, 5))
    plt.plot(sig_fft)
    #sig_fft -= sig_fft_noise

    power = np.abs(sig_fft)

    sample_freq = fftpack.fftfreq(sig.size, d=time_step)

    plt.figure(figsize=(6, 5))
    plt.plot(sample_freq, power)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('plower')

    high_freq_fft = sig_fft.copy()
    filtered_sig = fftpack.ifft(high_freq_fft*10)
    filtered_sig_ori = fftpack.ifft(sig_fft)

    plt.figure(figsize=(6, 5))
    plt.plot(sig, label='Original signal')
    plt.plot(filtered_sig, linewidth=3, label='Filtered signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    #filtered_sig.real filtered_sig.astype('float64') filtered_sig.astype('int16') filtered_sig.astype('int32')
    wavfile.write("highfrq.wav", samplerate, filtered_sig.astype('int32'))
    wavfile.write("orignal.wav", samplerate, filtered_sig_ori.astype('int32'))

    #plt.show()


def makeDataset(): #make nosie canceling
    time_step = 0.02
    samplerate = 44100
    period = 5.

    time_vec = np.arange(0, 20, time_step)

    spf = wave.open("test_3.wav", "r")
    signal = spf.readframes(-1)
    sig = np.fromstring(signal, "Int32") #Int32

    sig_fft = fftpack.fft(sig)

    print(len(sig))
    c = len(sig) + (500 - (len(sig) % 500))

    print(c)
    print(c/500)
    
    dataframe = pd.DataFrame(sig)
    dataframe.to_csv("./sig.csv", header = False, index = False)

    for i in range(2):
        spf_noise = wave.open(str(i)+".wav", "r")
        signal_noise = spf_noise.readframes(-1)
        sig_noise = np.fromstring(signal_noise, "Int32") #Int32
        
        dataframe = pd.DataFrame(sig_noise)
        dataframe.to_csv("./sig_noise_"+str(i)+".csv", header = False, index = False)

    '''
        result = np.zeros(sig.shape)
        for j in range(len(sig_noise)):
            result[j] = sig_noise[j]

        sig_fft_noise = fftpack.fft(result)
        sig_fft = sig_fft - sig_fft_noise
        #result = np.zeros(sig_fft.shape)
        #for j in range(len(sig_fft_noise)):
        #    result[j] = sig_fft_noise[j]
        
    '''
        
    '''
    for i in range(0, all_size, len(sig_noise)):
        print(str(a) + " : " + str(count))
        count += 1
        
        result_tmp =np.zeros(sig_noise.shape)
        
        for j in range(len(sig_noise)):
            sig[i+j] -= sig_noise[j]
            
            #result_tmp[j] = sig[i+j]
            
        result_tmp_fft = fftpack.fft(sig)
        #_sig = fftpack.ifft(result_tmp_fft - sig_fft_noise)
        np.append(result,result_tmp_fft)
    print(max(result))
    '''
    '''
    #_sig = fftpack.ifft(sig_fft[:5])
    sig_1 = sig[7000:7500]
    wavfile.write("highfrq_1.wav", samplerate, sig_1.astype('int32'))#int32

    sig_2 = sig[33500:34000]
    wavfile.write("highfrq_2.wav", samplerate, sig_2.astype('int32'))#int32

    sig_3 = sig[35500:36000]
    wavfile.write("highfrq_3.wav", samplerate, sig_3.astype('int32'))#int32

    sig_4 = sig[86500:87000]
    wavfile.write("highfrq_4.wav", samplerate, sig_4.astype('int32'))#int32

    sig_5 = sig[87500:88000]
    wavfile.write("highfrq_5.wav", samplerate, sig_5.astype('int32'))#int32

    sig_6 = sig[88000:88500]
    wavfile.write("highfrq_6.wav", samplerate, sig_6.astype('int32'))#int32

    sig_7 = sig[113000:113500]
    wavfile.write("highfrq_7.wav", samplerate, sig_7.astype('int32'))

    sig_8 = sig[115000:115500]
    wavfile.write("highfrq_8.wav", samplerate, sig_8.astype('int32'))

    sig_9 = sig[167500:168000]
    wavfile.write("highfrq_9.wav", samplerate, sig_9.astype('int32'))

    sig_10 = sig[168000:168500]
    wavfile.write("highfrq_10.wav", samplerate, sig_10.astype('int32'))

    sig_11 = sig[194500:195000]
    wavfile.write("highfrq_11.wav", samplerate, sig_11.astype('int32'))
    '''


def test():
    spf = wave.open("test_3.wav", "r")
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, "Int16")

    samplerate = 44100
    #samplerate = 64
    N = samplerate
    T = 1 / samplerate
    #T = 1.0 / 44100.0

    yf = fft(signal, N)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    #test_del = ifft(yf)
    yf[np.abs(N) > 1] = 0
    filtered_sig = ifft(yf)

    plt.figure(1)
    plt.subplot(311)
    plt.title("Signal Wave...")
    plt.plot(signal)

    plt.subplot(312)
    plt.title("Signal Wave...")
    plt.plot(filtered_sig)
    
    plt.subplot(313)
    plt.stem(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.xticks(xf, rotation=40)

    plt.show()

def analysis():
    train_audio_path = './'
    filename = 'noise.wav'
    samples, sample_rate = librosa.load(str(train_audio_path)+filename)

    N = 64
    T = 1.0 / 44100.0
    yf = fft(samples, N)
    
    #dataframe = pd.DataFrame(yf)
    #dataframe.to_csv("./fft.csv", header = False, index = False)

    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    plt.stem(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.xticks(xf, rotation=40)
    plt.show()
    #plt.savefig('test.png')

    #librosa.output.write_wav('test_nosie.wav', samples, sample_rate)


def del_noise():
    time_step = 0.02
    
    train_audio_path = './'
    filename_1 = 'test_3.wav'
    samples_1, sample_rate_1 = librosa.load(str(train_audio_path)+filename_1)

    filename_2 = 'noise.wav'
    samples_2, sample_rate_2 = librosa.load(str(train_audio_path)+filename_2)

    #samples_2 = -samples_2 ##삭제할 음원w
    '''
    print(len(samples_1))
    print(len(samples_2))
    
    if(len(samples_1) != len(samples_2)):
        mul = len(samples_1) - len(samples_2)
        print(mul)
        if(mul < 0):
            for i in range(len(samples_1)):
                samples_2[i] += samples_1[i]
                
            samples = samples_2
            sample_rate = sample_rate_2
        
        elif(mul > 0):
            for i in range(len(samples_2)):
                samples_1[i] += samples_2[i]
            samples = samples_1
            sample_rate = sample_rate_1

    else:
        samples_1 += samples_2
    '''
    #N = sample_rate
    N = sample_rate_1
    T = 1.0 / 44100.0
    
    #yf = fft(samples, sample_rate)
    yf_1 = fft(samples_1, N)
    yf_2 = fft(samples_2, N)

    #yf = np.abs(yf_1[0:N//2] - yf_2[0:N//2])
    #yf = yf_1 - yf_2

    test_del = ifft(yf_1)
    
    #print(test_del)
    #print(samples_1)

    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    '''
    plt.subplot(511)
    plt.plot(yf_1)
    
    plt.subplot(512)
    plt.plot(yf_2)
    
    plt.subplot(513)
    plt.plot(yf)
    
    plt.subplot(514)
    plt.plot(test_del)
    
    plt.subplot(515)
    plt.plot(samples_1)
    
    dataframe = pd.DataFrame(samples_1)
    dataframe.to_csv("./fft.csv", header = False, index = False)

    dataframe = pd.DataFrame(filtered_sig)
    dataframe.to_csv("./ifft.csv", header = False, index = False)

    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    plt.subplot(311)
    plt.stem(xf, 2.0/N * np.abs(yf_1[0:N//2]))
    #plt.xticks(xf, rotation=40)
    
    plt.subplot(312)
    plt.stem(xf, 2.0/N * np.abs(yf_2[0:N//2]))
    #plt.xticks(xf, rotation=40)
    
    plt.subplot(313)
    plt.stem(xf, 2.0/N * np.abs(yf[0:N//2]))
    #plt.xticks(xf, rotation=40)
 
    plt.savefig('test.png')
    '''
    librosa.output.write_wav('test_del_nosie.wav', samples_1, N)

    #plt.show()

#analysis()
#del_noise()
makeDataset()
#test2()
#test3()
