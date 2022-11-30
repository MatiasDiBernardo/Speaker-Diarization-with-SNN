#For metrics and testing of the model
from keras import Model
from keras.constraints  import unit_norm
from keras import backend as K
import librosa
import os
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Flatten, MaxPooling1D, Concatenate, Lambda, Input
from tensorflow.keras.layers import Dense, Conv1D
from sincnet_tensorflow import SincConv1D, LayerNorm

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def SiameseModel(input_dim):
        input_ref = Input((input_dim ,1), name="input_ref") # reference track
        input_dif = Input((input_dim ,1), name="input_dif") # different track

        sinc_layer1 = SincConv1D(N_filt=64,
                        Filt_dim=129,
                        fs=10000,
                        stride=16,
                        padding="SAME")

        sinc_layer2 = SincConv1D(N_filt=64,
                        Filt_dim=129,
                        fs=10000,
                        stride=16,
                        padding="SAME")
        #Stack A

        #a_1 = sinc_layer()
        a_2 = LayerNorm()

        a_3 = LeakyReLU(alpha=0.2)
        a_4 = MaxPooling1D(pool_size=2)


        a_5 = Conv1D(64, 3, strides=1, padding='valid')
        a_6 = BatchNormalization(momentum=0.05)
        a_7 = LeakyReLU(alpha=0.2)
        a_8 = MaxPooling1D(pool_size=2)

        a_9 = Conv1D(64, 3, strides=1, padding='valid')
        a_10 = BatchNormalization(momentum=0.05)
        a_11 = LeakyReLU(alpha=0.2)
        a_12 = MaxPooling1D(pool_size=2)

        a_13 = Conv1D(128, 3, strides=1, padding='valid')
        a_14 = BatchNormalization(momentum=0.05)
        a_15 = LeakyReLU(alpha=0.2)
        a_16 = MaxPooling1D(pool_size=2)

        a_17 = Conv1D(128, 3, strides=1, padding='valid')
        a_18 = BatchNormalization(momentum=0.05)
        a_19 = LeakyReLU(alpha=0.2)
        a_20 = MaxPooling1D(pool_size=2)

        a_21 = Flatten()

        #Stack B

        #b_1 = sinc_layer()
        b_2 = LayerNorm()

        b_3 = LeakyReLU(alpha=0.2)
        b_4 = MaxPooling1D(pool_size=2)


        b_5 = Conv1D(64, 3, strides=1, padding='valid')
        b_6 = BatchNormalization(momentum=0.05)
        b_7 = LeakyReLU(alpha=0.2)
        b_8 = MaxPooling1D(pool_size=2)

        b_9 = Conv1D(64, 3, strides=1, padding='valid')
        b_10 = BatchNormalization(momentum=0.05)
        b_11 = LeakyReLU(alpha=0.2)
        b_12 = MaxPooling1D(pool_size=2)

        b_13 = Conv1D(128, 3, strides=1, padding='valid')
        b_14 = BatchNormalization(momentum=0.05)
        b_15 = LeakyReLU(alpha=0.2)
        b_16 = MaxPooling1D(pool_size=2)

        b_17 = Conv1D(128, 3, strides=1, padding='valid')
        b_18 = BatchNormalization(momentum=0.05)
        b_19 = LeakyReLU(alpha=0.2)
        b_20 = MaxPooling1D(pool_size=2)

        b_21 = Flatten()

        mrg = Concatenate(axis=1, name="concatenate")

        # --- Fully connected layer => learned representation layer
        hidden_layer = Dense(256, activation="elu", kernel_constraint=unit_norm())
        
        # --- function to assemble shared layers
        def get_shared_dnn(m_input):
            shared_cnn_a = a_21(a_20(a_19(a_18(a_17(a_16(a_15(a_14(a_13(a_12(a_11(a_10(a_9(a_8(a_7(a_6(a_5(a_4(a_3(a_2(sinc_layer1(m_input)))))))))))))))))))))
            shared_cnn_b = b_21(a_20(b_19(b_18(b_17(b_16(b_15(b_14(b_13(b_12(b_11(b_10(b_9(b_8(b_7(b_6(b_5(b_4(b_3(b_2(sinc_layer2(m_input)))))))))))))))))))))

            return hidden_layer(mrg([shared_cnn_a,shared_cnn_b]))

        # --- instantiate shared layers
        siamese_ref = get_shared_dnn(input_ref)
        siamese_dif = get_shared_dnn(input_dif)

        # --- calculate dissimilarity
        dist  = Lambda(euclidean_distance, output_shape=lambda x: x[0])([siamese_ref, siamese_dif])
        
        # --- build model
        model = Model(inputs=[input_ref, input_dif], outputs=dist)
        
        return model

def test(test1_path, test2_path, model_path, fs, win, postive):
    audios_par = os.listdir(test1_path)
    audios_impar = os.listdir(test2_path)
    par1 = librosa.load(os.path.join(test1_path, audios_par[0]), sr=fs)
    par2 = librosa.load(os.path.join(test1_path, audios_par[1]), sr=fs)

    dif = librosa.load(os.path.join(test2_path, audios_impar[0]), sr=fs)

    par1 = par1[0:fs*win]
    par2 = par2[0:fs*win]
    dif = dif[0:fs*win]

    if postive:
        test = [par1, par2]
    else:
        test = [par1,dif]

    model = SiameseModel([10000, 10000])
    model.load_weights(model_path)

    pred = model.predict(test)

    print(pred)

test1 = 'data\Ordered Data\Pairs'
test2 = 'data\Ordered Data\Diff'
model_path = 'models\W_1seg_FS_10000_EP_10.h5'
fs = 10000
win = 1
pos = True

test(test1, test2, model_path, fs, win, pos)

