import librosa
import numpy as np
import scipy
import random
from keras import Sequential

#Extract x-vector of audio, ver si guardar toda la data en memoria o usar un generator

class FeatureExtraction():
    '''
    Extract the relevant frecuency information from the audio
    win_time: Interval of time to analize audio
    mel_coeff: Amount of mel coefficient to take into account
    filter: If we want or not pre-processing to the audio
    '''
    def __init__(self, win_time, mel_coeff, filter):

        self.win = win_time
        self.mel_coeff = mel_coeff
        self.filter = filter

    def check_length_and_pad(self, audio, fs):
        """
        Takes the audio and zero pads to the length
        defined by the window
        """

        if len(audio)/fs < self.win:
            diff = int(self.win * fs) - len(audio)//fs

            random_diff = random.randint(0, diff - 1)

            return np.hstack([np.zeros(diff - random_diff), audio, np.zeros(random_diff)])
        
        else:
            return audio[:fs*self.win]

    def filter_signal(self, audio, fs):
        """
        Apply filter to the signal to analize only the relevant
        frecuencies for speakers
        """

        #Hacer pasa alto y demás 

        pass



    def get_mel(self):
        pass

class X_Vector():
    '''
    Get the x vector from the audio
    '''

    def __init__(self):
        pass

    def conv_model(self):
        pass

    def get_xvector(self):
        pass