import os
import itertools
import pathlib
import shutil
import random

class LoadData():
    """
    Prepare the data from LibriSpeechASR.
    Move and regroup the audios.

    path_data: Directory with the folders with the speakers
    path_save: Place to save the reorderer data
    num_data: Max data pair for each speaker

    """
    def __init__(self, path_data, path_save, num_data):
        self.path_data = path_data
        self.path_save = path_save
        self.num_data = num_data

    def all_combinations(self, list_audio):
        """
        Estabilsh all the diferent combinations of the audio
        list with one speaker to create positive pairs
        """
        positive_pairs = []

        for pair in itertools.combinations(list_audio, 2):  #No incuyente
            positive_pairs.append(pair)

        return positive_pairs

    def combination_random(self, list_speakers):
        """
        From all the avaliable speakers choose two samples
        randomly to create negative pairs
        """
        n_max = self.num_data * len(list_speakers)
        negative_pairs = []

        for _ in range(n_max):
            a = random.randint(0, len(list_speakers) - 1)
            b = random.randint(0, len(list_speakers) - 1)

            spk1 = self.get_random_audio_from_speaker(list_speakers[a])
            spk2 = self.get_random_audio_from_speaker(list_speakers[b])

            negative_pairs.append([spk1, spk2])

        return negative_pairs

    def get_random_audio_from_speaker(self, dir_speaker):

        path = os.path.join(self.path_data, dir_speaker)
        only_first_folder = os.listdir(path)[0]
        first_folder_path = os.path.join(path, only_first_folder)
        audios_from_speaker = os.listdir(first_folder_path)

        pick = random.randint(0, len(audios_from_speaker) - 1)

        if audios_from_speaker[pick].split('.')[-1] == 'flac':
            return os.path.join(first_folder_path, audios_from_speaker[pick]) 
        else:
            return os.path.join(first_folder_path, audios_from_speaker[pick - 1]) 

    def string_num(self, i):
        num = str(i)

        if len(num) == 1:
            return '00' + num

        if len(num) == 2:
            return '0' + num

        return num

    def enter_first_folder(self, dir_speaker):
        #For now making the pairs only with the first folder
        path = os.path.join(self.path_data, dir_speaker)
        only_first_folder = os.listdir(path)[0]
        first_folder_path = os.path.join(path, only_first_folder)

        return list(pathlib.Path(first_folder_path).glob('*.flac'))  #Para sacar los txt

    def positive_pairs(self):
        """
        Create dir_pairs, nested list where the first index is the
        speakar and the second is a positive pair
        """
        speakers_list = os.listdir(self.path_data)
        dir_pairs = []

        for speaker in speakers_list:
            audios_per_speaker = self.enter_first_folder(speaker)
            dir_pairs.append(self.all_combinations(audios_per_speaker))
        
        return dir_pairs

    def negative_pairs(self):
        """
        Create dir_diff with random combinations of pairs with 
        different speakers
        """
        speaker_list = os.listdir(self.path_data)
        return self.combination_random(speaker_list)

    def move_file_pair(self, num_speaker, par_number, par):
        new_name = f'S{self.string_num(num_speaker)}_{self.string_num(par_number)}'
        save = os.path.join(self.path_save, 'Pairs')  #Crear carpeta

        shutil.copy(par[0], save)
        shutil.copy(par[1], save)

        os.rename(os.path.join(save, par[0].name), os.path.join(save, new_name + '-1.flac'))
        os.rename(os.path.join(save, par[1].name), os.path.join(save, new_name + '-2.flac'))

    def move_file_diff(self, num, par):
        new_name = f'D_{self.string_num(num)}'
        save = os.path.join(self.path_save, 'Diff')  #Crear carpeta

        if par[0] != par[1]:
            shutil.copy(par[0], save)
            shutil.copy(par[1], save)

            os.rename(os.path.join(save, par[0].split('\\')[-1]), os.path.join(save, new_name + '-1.flac'))
            os.rename(os.path.join(save, par[1].split('\\')[-1]), os.path.join(save, new_name + '-2.flac'))

    def create_pairs(self):
        
        dir_pairs = self.positive_pairs()

        for i in range(len(dir_pairs)):  #i is num of speaker
            speaker = dir_pairs[i]
            for j in range(len(speaker)):  #j is par number
                if j < self.num_data:
                    par = speaker[j]
                    self.move_file_pair(i, j, par)
                else:
                    continue
    
    def create_diff(self):

        dir_diff = self.negative_pairs()

        for i in range(len(dir_diff)):
            self.move_file_diff(i, dir_diff[i])

    def create_data(self):
        self.create_pairs()
        self.create_diff()


path_data = 'data\LibriSpeech\dev-clean'
path_save = 'data\Ordered Data'
num_pairs = 100

test = LoadData(path_data, path_save, num_pairs)
#test.create_data()