import librosa
import numpy as np
import scipy
from keras import Sequential

#Extract x-vector of audio, ver si guardar toda la data en memoria o usar un generator

class FeatureExtraction():
    '''
    Extract the relevant frecuency information from the audio
    '''
    def __init__(self):
        pass

    def check_length_and_pad(self, audio, fs):
        """
        Takes the audio and zero pads to the length
        defined by the window
        """

        if len(audio)/fs < self.win:
            diff = self.win * fs - len(audio)//fs

            return np.hstack([np.zeros(diff), audio])  #Hacer que sea adelante o atras random
        
        else:
            return audio[:fs*self.win]

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