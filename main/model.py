import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Flatten, MaxPooling1D, Concatenate, Lambda, Input
from keras.regularizers import l2
from keras.constraints  import unit_norm
from keras.models       import Model
from keras import backend as K
from sincnet_tensorflow import SincConv1D, LayerNorm


#Siamese network to identify similarity between speakers

class Siamese_Network():

    def __init__(self, input, output, batch_size):
        self.input = input
        self.output = output
        self.batch_size = batch_size
    
    def euclidean_distance(self, vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    def SincNet(self, input_red):

        sinc_layer = SincConv1D(N_filt=64,
                        Filt_dim=129,
                        fs=16000,
                        stride=16,
                        padding="SAME")

        x = sinc_layer(input_red)
        x = LayerNorm()(x)

        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling1D(pool_size=2)(x)


        x = Conv1D(64, 3, strides=1, padding='valid')(x)
        x = BatchNormalization(momentum=0.05)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling1D(pool_size=2)(x)

        x = Conv1D(64, 3, strides=1, padding='valid')(x)
        x = BatchNormalization(momentum=0.05)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling1D(pool_size=2)(x)

        x = Conv1D(128, 3, strides=1, padding='valid')(x)
        x = BatchNormalization(momentum=0.05)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling1D(pool_size=2)(x)

        x = Conv1D(128, 3, strides=1, padding='valid')(x)
        x = BatchNormalization(momentum=0.05)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling1D(pool_size=2)(x)

        x = Flatten()(x)

        return x

    def model(self):
        
        input_ref = Input((self.input ,1), name="input_ref") # reference track
        input_dif = Input((self.input ,1), name="input_dif") # different track

        # --- merge parallel CNN Stacks
        mrg = Concatenate(axis=1, name="concatenate")

        # --- Fully connected layer => learned representation layer
        hidden_layer = Dense(256, activation="elu", kernel_constraint=unit_norm())
        
        # --- function to assemble shared layers
        def get_shared_dnn(m_input):
            shared_cnn_a = self.SincNet(m_input)  #No se si tiene que ser instancias separadas
            shared_cnn_b = self.SincNet(m_input)

            return hidden_layer(mrg([shared_cnn_a,shared_cnn_b]))

        # --- instantiate shared layers
        siamese_ref = get_shared_dnn(input_ref)
        siamese_dif = get_shared_dnn(input_dif)

        # --- calculate dissimilarity
        dist  = Lambda(self.euclidean_distance, output_shape=lambda x: x[0])([siamese_ref, siamese_dif])
        
        # --- build model
        model = Model(inputs=[input_ref, input_dif], outputs=dist)
        
        return model

test_red = Siamese_Network(16000, 100, 126)

model = test_red.model()
model.summary()