# tnet.py
import numpy as np
import sys

from keras import backend as K
from keras.layers import Input, merge, Dense, ELU
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.engine.topology import Layer

from tnet_generator_multi import Generator

L2_COEFF = 0.0001

class Tnet(Layer):
    def __init__(self, K1, K2, D, norm, **kwargs):
        self.D = D
        self.K1 = K1
        self.K2 = K2
        self.normalize = norm
        super(Tnet, self).__init__(**kwargs)

    def tlayer(self, O, T, W, P, K, D, b):
        if self.normalize:
            O = K.batch_normalization(x=O, mean=K.mean(O), var=K.var(O), gamma=1., beta=0., epsilon=0.0001)
            P = K.batch_normalization(x=P, mean=K.mean(P), var=K.var(P), gamma=1., beta=0., epsilon=0.0001)

        T_ = K.reshape(T, [D, D * K])
        OT = K.dot(O, T_)
        OT = K.reshape(OT, [-1, D, K])

        P_ = K.reshape(P, [-1, D, 1])
        OTP = K.batch_dot(OT, P_, axes=(1, 1))

        OP = K.concatenate([O, P], axis=1)
        W_ = K.transpose(W)

        WOP = K.dot(OP, W_)
        WOP = K.reshape(WOP, [-1, K, 1])

        b_ = K.reshape(b, [K, 1])

        S = merge([OTP, WOP, b_], mode='sum')
        S_ = K.reshape(S, [-1, K])

        R = K.tanh(S_)

        return R

    def build(self, input_shape):
        D = self.D
        K1 = self.K1
        K2 = self.K2

        self.T1 = K.variable(np.random.uniform(low=-1, high=1, size=(D, D, K1)))
        self.T2 = K.variable(np.random.uniform(low=-1, high=1, size=(D, D, K1)))
        self.T3 = K.variable(np.random.uniform(low=-1, high=1, size=(K1, K1, K2)))
        self.W = K.variable(np.random.uniform(low=-1, high=1, size=(K1, 2 * D)))
        self.b = K.variable(np.zeros((K1,)))

        self.trainable_weights = [self.T1, self.T2, self.T3, self.W, self.b]
        self.regularizers = [self.get_regularizer(x) for x in self.trainable_weights]

    def call(self, x, mask=None):
        O, P = x
        name = O.name[:-2]

        tensor_weights = {'O1': self.T1, 'O2': self.T2, 'P': self.T2,
                          'C1': self.T1, 'C2': self.T1, 'C3': self.T1, 'C4': self.T1, 'C5': self.T1,
                          'Tanh': self.T3, 'Tanh_2': self.T3, 'Tanh_3': self.T3, 'Tanh_4': self.T3,
                          'Tanh_5': self.T3, 'Tanh_6': self.T3, 'Tanh_7': self.T3}
        T = tensor_weights[name]

        if name in ['Tanh', 'Tanh_2', 'Tanh_3', 'Tanh_4', 'Tanh_5', 'Tanh_6', 'Tanh_7']:
            D = self.K1
            K = self.K2
            W = K.sum(K.reshape(self.W, [-1, K, D * 2]), axis=0)
            b = K.sum(K.reshape(self.b, [-1, K]), axis=0)
            self.level = 2
        else:
            D = self.D
            K = self.K1
            W = self.W
            b = self.b
            self.level = 1

        return self.tlayer(O=O, P=P, T=T, W=W, K=K, D=D, b=b)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        K = self.K1 if self.level == 1 else self.K2
        return (batch_size, K)

    def get_regularizer(self, W):
        reg = l2(L2_COEFF)
        reg.set_param(W)
        return reg

def make_model(train):
    D = 100
    K1 = 10
    K2 = 5
    N = 500  # batch size

    tnet = Tnet(K1=K1, K2=K2, D=D, norm=train, batch_input_shape=[N, D])

    O1 = Input((D,), name='O1')
    O2 = Input((D,), name='O2')
    P = Input((D,), name='P')

    C1 = Input((D,), name='C1')
    C2 = Input((D,), name='C2')
    C3 = Input((D,), name='C3')
    C4 = Input((D,), name='C4')
    C5 = Input((D,), name='C5')

    R1 = tnet([O1, P])
    R2 = tnet([P, O2])
    R1_ = tnet([C1, P])
    R2_ = tnet([C2, P])
    R3_ = tnet([C3, P])
    R4_ = tnet([C4, P])
    R5_ = tnet([C5, P])

    U = tnet([R1, R2])
    U_1 = tnet([R1_, R2])
    U_2 = tnet([R2_, R2])
    U_3 = tnet([R3_, R2])
    U_4 = tnet([R4_, R2])
    U_5 = tnet([R5_, R2])

    out = merge([U, U_1, U_2, U_3, U_4, U_5], mode='concat', concat_axis=-1)

    if train:
        model = Model(inputs=[O1, P, O2, C1, C2, C3, C4, C5], outputs=out)
    else:
        model = Model(inputs=[O1, P, O2], outputs=U)

    return model

def margin_loss(y_true, y_pred):
    i = int(K.int_shape(y_pred)[1] / 6)
    U = y_pred[:, :i]
    U_1 = y_pred[:, i:i + 1]
    U_2 = y_pred[:, i + 1:i + 2]
    U_3 = y_pred[:, i + 2:i + 3]
    U_4 = y_pred[:, i + 3:i + 4]
    U_5 = y_pred[:, i + 4:i + 5]

    corrupts = [U_1, U_2, U_3, U_4, U_5]

    loss = 0.
    for c in corrupts:
        loss += K.sum(K.maximum(0., 1. - U + c), axis=-1)
    return loss

def train(batch, epochs):
    model = make_model(train=True)
    generator = Generator(batch_size=batch)
    epoch_samples = generator.get_data_size() - (generator.get_data_size() % batch)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    checkpoint = ModelCheckpoint(filepath='./tnet-weights/checkpoint-{epoch:02d}.hdf5', save_weights_only=True)

    model.compile(optimizer=adam, loss=margin_loss, metrics=['accuracy'])
    model.fit_generator(generator.generate(), steps_per_epoch=epoch_samples // batch,
                        epochs=epochs, callbacks=[checkpoint], verbose=1)

def predict(weight):
    model = make_model(train=False)
    tuples = np.load('./data3/titles-encoded-tuples.npy')
    O1, P, O2 = tuples[:, 0], tuples[:, 1], tuples[:, 2]
    X = [O1, P, O2]

    model.load_weights(f'./tnet-weights/checkpoint-{weight}.hdf5')
    predictions = model.predict(x=X, verbose=1, batch_size=500)
    np.save('./data/event-embeddings.npy', predictions)
    print("Shape of output: ", predictions.shape)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python tnet.py <train/predict> [weight]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == 'train':
        train(batch=500, epochs=500)
    elif mode == 'predict':
        if len(sys.argv) < 3:
            print("Usage: python tnet.py predict <weight>")
            sys.exit(1)
        weight = sys.argv[2]
        predict(weight)
