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

    
    def load(self,file):
        filepath = os.path.join(self.dir,file)
        try:
            self.data_frame = pd.read_csv(filepath)
        except Exception as e:
            raise Exception("Can't load data csv file. Error: {}".format(e))

        self.num_samples  = self.data_frame.shape[0]
        self.num_features = self.data_frame.shape[1]-1

        self.X = np.zeros([self.num_samples,self.num_features])
        self.Y = np.zeros([self.num_samples,1])
        for index,row in self.data_frame.iterrows():
            self.X[index] = row.iloc[:-1].to_numpy()
            self.Y[index] = row.iloc[-1]
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self,idx):
        if not self.do_torch:
            return self.X[idx],self.Y[idx]
        else:
            return torch.as_tensor(self.X[idx],dtype=torch.float32),torch.as_tensor(self.Y[idx],dtype=torch.float32)    





        
