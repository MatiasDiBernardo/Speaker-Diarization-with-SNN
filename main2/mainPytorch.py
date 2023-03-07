import torch
import torch.nn as nn  
import torch.optim as optim
import torchaudio
import torch.nn.functional as F
import librosa
import random
import numpy as np
import os
import pandas as pd
from torch.utils.data import (
    Dataset,
    DataLoader,
)

class SpeakerDiarizationDataloader(Dataset):
    """
    Generates the matrix of features from the Wav2Vec model.
    root_par: Directory with the same speakers label
    root_dif: Directory with the different speakers label
    wav2vec: Pytorch object with the Wav2Vec model. 
    """
    def __init__(self, root_par, root_dif, wav2vec):
        self.root_par = root_par
        self.root_dif = root_dif
        self.wav2vec_model = wav2vec

    def __len__(self):
        return 3833

    def __getitem__(self, index):
        par_list = os.listdir(self.root_par)
        dif_list = os.listdir(self.root_dif)
        if index % 2 == 0:  #Same speaker
            audio1, audio2 = self.extractFeatures(self.root_par, par_list, index, 5.13)
            label = torch.tensor(int(0))

            return ((audio1, audio2), label)
        else:  #Different speaker
            audio1, audio2 = self.extractFeatures(self.root_dif, dif_list, index - 1, 5.13)
            label = torch.tensor(int(1))

            return ((audio1, audio2), label)

    def cutOrPad(self, audio, win, fs):
        if len(audio)/fs < win:
            diff = int(win * fs) - len(audio)  #In samples

            random_diff = random.randint(0, diff - 1)

            return np.hstack([np.zeros(diff - random_diff), audio, np.zeros(random_diff)])
        
        else:
            return audio[:int(fs*win)]
    
    def extractFeatures(self, root, list_audios, index, seg_to_cut):
        path1 = os.path.join(root, list_audios[index])
        path2 = os.path.join(root, list_audios[index + 1])

        sig1, fs1 = librosa.load(path1, sr=16000, mono=True)
        sig1 = self.cutOrPad(sig1, seg_to_cut, fs1)
        sig1 = torch.from_numpy(sig1)
        sig1 = sig1[None, :]
        #Feat 1 shape (1, 256, 512)
        feat1, length = self.wav2vec_model.feature_extractor(sig1.float(), sig1.shape[1])

        sig2, fs2 = librosa.load(path2, sr=16000, mono=True)
        sig2 = self.cutOrPad(sig2, seg_to_cut, fs2)
        sig2 = torch.from_numpy(sig2)
        sig2 = sig2[None, :]
        feat2, length2 = self.wav2vec_model.feature_extractor(sig2.float(), sig2.shape[1])

        return feat1, feat2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 3e-4
batch_size = 32
num_epochs = 10

#Load Wav2VecModel
bundle = torchaudio.pipelines.WAV2VEC2_BASE  #This is not fine tuned before
Wav2VecModel = bundle.get_model()
# freeze all layers, change final linear layer with num_classes
for param in Wav2VecModel.parameters():
    param.requires_grad = False

Wav2VecModel.to(device)

# Load Data
dataset = SpeakerDiarizationDataloader(
    root_par="data\Ordered Data\Pairs",
    root_dif="data\Ordered Data\Diff",
    wav2vec= Wav2VecModel,
)

train_set, test_set = torch.utils.data.random_split(dataset, [3066, 767])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

class SiameseNetwork(nn.Module):
    """
    2 paralel convolutional layer are applied to the output of the wav2vec model
    output, followed by a 3 dense layers to obtain the representative value of the
    audio speaker.
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(128, 256, kernel_size=5,stride=2),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,2)
        )
        
    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)  #Flatten
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

net = SiameseNetwork()
net.to(device)

#Constrastive Loss
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

      return loss_contrastive
     
# Loss and optimizer
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data1 = data[0].to(device=device)
        data2 = data[1].to(device=device)
        targets = targets.to(device=device)

        # forward
        au1feats, au2feats = net(data1, data2)

        loss = criterion(au1feats, au2feats, targets)
        print("El loss", loss)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")

# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for data, target in loader:
            feat1 = data[0].to(device=device)
            feat2 = data[1].to(device=device)
            target = target.to(device=device)

            output1, output2 = model(feat1, feat2)
            sim_value = F.pairwise_distance(output1, output2)
            if sim_value < 0.3 and target == 0:
                num_correct += 1
            
            if sim_value > 0.7 and target == 1:
                num_correct += 1

            num_samples += 1

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()

print("Checking accuracy on Training Set")
#check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
#check_accuracy(test_loader, model)

torch.save(net.state_dict(), "models\\pytorch_1.pt")
