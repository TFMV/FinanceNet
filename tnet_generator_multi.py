import random
import numpy as np
 
class Generator():
    def __init__(self, batch_size):
        self.data = np.load('./data/body-encoded-tuples.npy') #non-normalized
        # self.data = np.load('./data/body-encoded-tuples2.npy')  #normalized
        self.batch_size = batch_size
        self.y = np.zeros((batch_size, 4))
 
    def get_data_size(self):
        """
        Return total number of samples
        """
        return self.data.shape[0]
 
    def get_batch_size(self):
        """
        Return size of batches
        """
        return self.batch_size
 
    def generate(self):
        """
        Return a generator object for training
        """
        while 1:
            # Select input samples
            random_indices = random.sample(range(len(self.data)), self.batch_size)
            inputs = [(self.data[i][0], self.data[i][1], self.data[i][2]) for i in random_indices]
            O1 = np.array([x[0] for x in inputs])
            P = np.array([x[1] for x in inputs])
            O2 = np.array([x[2] for x in inputs])
 
            # Select corrupted element
            corrupts = []
            for _ in range(5):
                corrupt_indices = random.sample(range(len(self.data)), self.batch_size)
                corrupted =[self.data[i][0] for i in corrupt_indices]
                corrupts.append(np.array(corrupted))
 
            C1,C2,C3,C4,C5 = corrupts
 
            # Make sure all inputs have same number of samples
            shapes = [x for x in [O1.shape[0], P.shape[0], O2.shape[0], C1.shape[0], \
                                            C2.shape[0], C3.shape[0], C4.shape[0], C5.shape[0]] ]
            shape_check = set(shapes)
 
            # yield tensor net inputs
            if len(shape_check) == 1:
                yield ([O1, P, O2, C1, C2, C3, C4, C5], self.y)
