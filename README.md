# Speaker-Diarization-with-SNN
Final project for the Seminario en Aplicaciones de Redes Neuronales en la recuperación de información musical. The objetive is to use a Siamese Neuronal Network architecture in the Speaker Diarization task. We use the Librispeech dataset for training and validation.

## First Implementation
The first implementation is in Keras and uses the [SincNet](https://github.com/mravanelli/SincNet) architecture to lower the dimensionality of the convolutional task and work directly with the raw audio. With this approach we can obteain a good training error but the model does not generalize well and the validetion error was high.

## Second Implementation
The second implementation is in PyTorch and uses the [Wav2Vec](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) model to extract the acoustic features of raw audio and proceed with this low dimensionality vector for the analysis. 
