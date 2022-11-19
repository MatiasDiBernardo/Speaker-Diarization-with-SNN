import numpy as np
import pandas as pd
import os
import itertools
import pathlib
import shutil

#Prepare the raw dataset into the pairs spected by the model

class LoadData():
    """
    Prepare the data from LibriSpeechASR.
    Move and regroup the audios with pairs.

    path_data: Directory with the folders with the speakers
    path_save: Place to save the reorderer data
    num_data: Max data pair for speaker

    """
    def __init__(self, path_data, path_save, num_data, window):
        self.path_data = path_data
        self.path_save = path_save
        self.num_data = num_data

    def all_combinations(self, list_audio):
        """
        Estabilsh all the diferent combinations of the audio
        list with one speaker to create postive pairs
        """
        positive_pairs = []

        for pair in itertools.combinations(list_audio, 2):  #No incuyente
            positive_pairs.append(pair)

        return positive_pairs

    def enter_first_folder(self, dir_speaker):
        #For now making the pairs only with the first folder
        path = os.path.join(self.path_data, dir_speaker)
        only_first_folder = os.listdir(path)[0]
        first_folder_path = os.path.join(path, only_first_folder)
        #Si no funca esto borrarlos manual y listo
        return list(pathlib.Path(first_folder_path).glob('*.flac'))  #Para sacar los txt


    def positive_pairs(self):
        """
        Create dir_pairs, nested list where the first index is the
        speakar and the second is one pair
        """

        speakers_list = os.listdir(self.path_data)
        self.dir_pairs = []

        for speaker in speakers_list:
            audios_per_speaker = self.enter_first_folder(speaker)
            self.dir_pairs.append(self.all_combinations(audios_per_speaker))

    def negative_pairs(self):
        return None

    def move_file(self, num_speaker, par_number, par):
        new_name = f'S{str(num_speaker)}_{str(par_number)}'
        save = os.path.join(self.path_save, 'Pairs')  #Crear carpeta

        shutil.copy(par[0], save)
        shutil.copy(par[1], save)

        # par[0] es un objeto de pathlib y con name me da el archivo base

        #Con rename lo puedo hacer directo pero no me copia el original
        os.rename(os.path.join(save, par[0].name), os.path.join(save, new_name + '-1.flac'))
        os.rename(os.path.join(save, par[1].name), os.path.join(save, new_name + '-2.flac'))


    def move_ane_create_csv(self):
        "Mark the pairs with 0 if same speaker and 1 if diferent"
        self.positive_pairs()
        for i in range(len(self.dir_pairs)):  #i is num of speaker
            speaker = self.dir_pairs[i]
            for j in range(len(speaker)):  #j is par number
                par = speaker[j]
                self.move_file(i, j, par)

path_data = 'data\LibriSpeech\dev-clean'
path_save = 'data\Ordered Data'

ah = LoadData(path_data, path_save, 1000)

ah.move_ane_create_csv()