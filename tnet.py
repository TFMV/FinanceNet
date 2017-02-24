import numpy as np
import random
import sys
from tnet_generator_multi import Generator
 
from keras import backend as BE
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, merge
import keras.regularizers as regularizers
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
 
L2_COEFF = 0.0001
def get_regularizer(W):
    reg = regularizers.get(l2(L2_COEFF))
    reg.set_param(W)
    return reg
 
class Tnet(Layer):
    def __init__(self, K1, K2, D, norm, **kwargs):
        self.D = D
        self.K1 = K1
        self.K2 = K2
        self.normalize = norm
        super().__init__(**kwargs)
 
    def tlayer(self, O, T, W, P, K, D, b):
 
        if self.normalize:
            O = BE.batch_normalization(x=O, mean=BE.mean(O), var=BE.var(O), gamma=1., beta=0., epsilon=0.0001)
            P = BE.batch_normalization(x=P, mean=BE.mean(P), var=BE.var(P), gamma=1., beta=0., epsilon=0.0001)
 
        T_ = BE.reshape(T, [D, D * K])
        OT = BE.dot(O, T_)
        OT = BE.reshape(OT, [-1, D, K])
 
        P_ = BE.reshape(P, [-1, D, 1])
        OTP = BE.batch_dot(OT, P_, axes=(1,1))
 
        OP = BE.concatenate([O, P], axis=1)
        W_ = BE.transpose(W)
 
        WOP = BE.dot(OP, W_)
        WOP = BE.reshape(WOP, [-1, K, 1])
 
        b_ = BE.reshape(b, [K, 1])
 
        S = merge([OTP, WOP, b_], mode='sum')
        S_ = BE.reshape(S, [-1, K])
 
        R = BE.tanh(S_)
 
        # print('O shape: ', BE.int_shape(O))
        # print('T_ shape: ', BE.int_shape(T_))
        # print('OT shape:', BE.int_shape(OT))
        # print('P shape: ', BE.int_shape(P))
        # print('P_ shape: ', BE.int_shape(P_))
        # print('OTP shape:', BE.int_shape(OTP))
        # print('OP shape: ', BE.int_shape(OP))
        # print('WOP shape: ', BE.int_shape(WOP))
        # print('WOP reshape: ', BE.int_shape(WOP))
        # print('b_ shape: ', BE.int_shape(b_))
        # print('S shape: ', BE.int_shape(S))
        # print('S_ shape: ', BE.int_shape(S_))
 
        return R
 
    def build(self, input_shape):
        D = self.D
        K1 = self.K1
        K2 = self.K2
 
        self.T1 = BE.variable(np.random.uniform(low=-1, high=1, size=(D, D, K1) ))
        self.T2 = BE.variable(np.random.uniform(low=-1, high=1, size=(D, D, K1) ))
        self.T3 = BE.variable(np.random.uniform(low=-1, high=1, size=(K1, K1, K2) ))
        self.W = BE.variable(np.random.uniform(low=-1, high=1, size=(K1, 2 * D) ))
        self.b = BE.variable(np.zeros((K1, ) ) )
 
        self.trainable_weights = [self.T1, self.T2, self.T3, self.W, self.b]
        self.regularizers = ([get_regularizer(x) for x in self.trainable_weights])
 
    def call(self, x, mask=None):
        O, P = x
        name = O.name[:-2]
 
        # Use the tensor parameter according to position in the network
        tensor_weights = {'O1':self.T1, 'O2':self.T2, 'P':self.T2, \
                                    'C1':self.T1, 'C2': self.T1,'C3':self.T1, 'C4':self.T1, 'C5':self.T1, \
                                    'Tanh':self.T3, 'Tanh_2':self.T3,'Tanh_3':self.T3,'Tanh_4':self.T3, \
                                    'Tanh_2':self.T3,'Tanh_5':self.T3, 'Tanh_6':self.T3, 'Tanh_7':self.T3}
        T = tensor_weights[name]
 
        # Inner branch takes in K input size (output size of previous branches)
        if name in ['Tanh', 'Tanh_2', 'Tanh_3', 'Tanh_4', 'Tanh_5', 'Tanh_6', 'Tanh_7']:
            D = self.K1
            K = self.K2
            W = BE.sum(BE.reshape(self.W, [-1, K, D*2]), axis=0)
            b = BE.sum(BE.reshape(self.b, [-1, K]), axis=0)
            self.level = 2
        else:
            D = self.D
            K = self.K1
            W = self.W
            b = self.b
            self.level = 1
 
        out = self.tlayer(O=O, P=P, T=T, W=W, K=K, D=D, b=b)
        return out
 
    def get_output_shape_for(self, input_shape):
        batch_size = input_shape[0]
        K = self.K1 if self.level ==1 else self.K2
        return (batch_size, K)
 
def make_model(train):
    D = 100
    K1 = 10
    K2 = 5
    N = 500 #batch size
 
    tnet = Tnet(K1=K1, K2=K2, D=D, norm=train, batch_input_shape=[N, D])
 
    # Tuple mini-batches
    O1 = Input((D,), name='O1')
    O2 = Input((D,), name='O2')
    P = Input((D,), name='P')
 
    # Corrupt example mini-batches
    C1 = Input((D,), name='C1')
    C2 = Input((D,), name='C2')
    C3 = Input((D,), name='C3')
    C4 = Input((D,), name='C4')
    C5 = Input((D,), name='C5')
 
    # Outer brach calculation
    R1 =  tnet([O1, P])     # R1 name = 'Tanh'
    R2 =  tnet([P, O2])     # R2 name = 'Tanh_2'
    R1_ =  tnet([C1, P])
    R2_ =  tnet([C2, P])
    R3_ =  tnet([C3, P])
    R4_ =  tnet([C4, P])
    R5_ =  tnet([C5, P])
 
    # Inner branch calculation
    U = tnet([R1, R2])
    U_1 = tnet([R1_, R2])
    U_2 = tnet([R2_, R2])
    U_3 = tnet([R3_, R2])
    U_4 = tnet([R4_, R2])
    U_5 = tnet([R5_, R2])
 
    out = merge([U, U_1, U_2, U_3, U_4, U_5], mode='concat', concat_axis=-1)
 
    if train:
        model = Model(input=[O1, P, O2, C1, C2, C3, C4, C5], output=out)
 
    else:
        model = Model(input=[O1, P, O2], output=U)
 
    return model
 
def margin_loss(y_true, y_pred):
    # Unsplit tensors
    i = int(BE.int_shape(y_pred)[1] / 6)
    U = y_pred[:,:i]        # Splits in half on axis 1
    U_1 = y_pred[:,i:i+1]
    U_2 = y_pred[:,i+1:i+2]
    U_3 = y_pred[:,i+2:i+3]
    U_4 = y_pred[:,i+3:i+4]
    U_5 = y_pred[:,i+4:i+5]
 
    corrupts = [U_1, U_2, U_3, U_4, U_5]
 
    # Calculate loss
    loss = 0.
    for c in corrupts:
        loss += BE.sum(BE.maximum(0., 1. - U + c), axis=-1)
    return loss
 
def train(batch, epochs):
    model = make_model(train=True)
    # model.summary(line_length=150)
 
    generator = Generator(batch_size=batch)
    epoch_samples = generator.get_data_size() - (generator.get_data_size()%batch)
 
    # adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0)
    # adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=0.5)
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
 
    # model.load_weights('./tensornet-weights/cont/checkpoint-174.hdf5')
 
    checkpoint = ModelCheckpoint(filepath='./tensornet-weights/final/checkpoint-{epoch:02d}.hdf5', save_weights_only=True)
    model.compile(optimizer=adam, loss=margin_loss, metrics=['accuracy'])
    model.fit_generator(generator.generate(), samples_per_epoch=epoch_samples, \
                                    nb_epoch=epochs, callbacks=[checkpoint], verbose=1)
 
def predict(weight):
    model = make_model(train=False)
    # tuples = np.load('./data/titles-encoded-tuples.npy')        #Non-normalized
    tuples = np.load('./data3/titles-encoded-tuples.npy')
    O1, P, O2 = tuples[:,0], tuples[:,1], tuples[:,2]
    X = [O1, P, O2]
 
    model.load_weights('./tensornet-weights-server/checkpoint-{}.hdf5'.format(weight))
    # model.load_weights('./tensornet-weights/multi/cont/checkpoint-{}.hdf5'.format(weight))
    predictions = model.predict(x=X, verbose=1, batch_size=500)
    np.save('./data/event-embeddings.npy', predictions)
    print("Shape of output: ", predictions.shape)
 
if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train(batch=500, epochs=500)
    elif sys.argv[1] == 'predict':
        predict(sys.argv[2])