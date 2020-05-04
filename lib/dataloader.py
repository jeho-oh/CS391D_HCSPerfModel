import os
import pandas as pd
import numpy as np
import torch


# DataLoader in pytorch's style
class DataLoader:
    def __init__(self,datadir,do_torch=False):
        self.dir = datadir
        self.X = None
        self.Y = None
        self.data_frame = None
        self.do_torch = do_torch 
        self.num_features = 0
        self.num_samples = 0 

    
    def load(self,file,stack_features=0,scale=False):
        filepath = os.path.join(self.dir,file)
        try:
            self.data_frame = pd.read_csv(filepath)
        except Exception as e:
            raise Exception("Can't load data csv file. Error: {}".format(e))

        self.num_samples  = self.data_frame.shape[0]
        if not stack_features:
            self.num_features = self.data_frame.shape[1]-1
        else:
            self.num_features = int((self.data_frame.shape[1]-1)/
                                     stack_features)+1
            powers_of_two = np.repeat(np.arange(stack_features).reshape(1,-1),
                                      self.num_features, axis=0)

            powers_mat = np.power(2,powers_of_two)

        
        self.X = np.zeros([self.num_samples,self.num_features])
        self.Y = np.zeros([self.num_samples])
        for index,row in self.data_frame.iterrows():
            self.Y[index] = row.iloc[-1]
            if not stack_features:
                self.X[index] = row.iloc[:-1].to_numpy()
                
            else:
                features = row.iloc[:-1]\
                                 .to_numpy().ravel()
                features.resize(self.num_features*stack_features)                                               
                features = features.reshape(self.num_features,stack_features)
                
                self.X[index] = np.sum(np.multiply(features,powers_mat),
                                       axis=1)

        
        np.random.seed(0)
        shuffled_indexes = np.random.permutation(np.arange(0,self.num_samples))
        self.Y = self.Y[shuffled_indexes]
        self.X = self.X[shuffled_indexes]
        import time
        t = 1000 * time.time() # current time in milliseconds
        np.random.seed(int(t) % 2**32)
        print(self.num_features)
        print(self.X[0])

        if scale:
            # Scale X_train and Y_train
            max_X = np.amax(self.X, axis=0)
            if 0 in max_X:
                max_X[max_X == 0] = 1
            self.X = np.divide(self.X, max_X)
            max_Y = np.max(self.Y)/100
            if max_Y == 0:
                max_Y = 1
            self.Y = np.divide(self.Y, max_Y)

        print("dtype:" + str(self.X.dtype))
    def __len__(self):
        return self.num_samples

    def __getitem__(self,idx):
        if not self.do_torch:
            return self.X[idx],self.Y[idx]
        else:
            return torch.as_tensor(self.X[idx],dtype=torch.float32),\
                   torch.as_tensor(self.Y[idx],dtype=torch.float32)    





        
