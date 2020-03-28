#!/usr/bin/python3

# Add ../lib to path.
import os
bindir = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.join(bindir,'..','lib',''))

# Python modules
import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np

## Local modules from ../lib
from nn_experimental import ExperimentalNN
from utils import logmsg, openlog
from dataloader import DataLoader

DATA_DIR = os.path.join(bindir,'..','Data','')


def parse_args(argv):
    argParser = argparse.ArgumentParser(
                          formatter_class=argparse.RawTextHelpFormatter, 
                          description = "experimental runner for NN classifier")
    argParser.add_argument('-log', action = "store", dest = "logfile",
                           required = False, help = "logname",default=None)
    argParser.add_argument('-dataset', action = "store", dest = "dataset",
                           required = False, help = "csv data file to load",
                           default=None)
    return argParser.parse_args(argv)
    

def calc_mse_loss(X,Y,nn_model,pprint=False):
    sample_num = X.shape[0] 
    Y_pred = torch.zeros(sample_num)
    loss_per_sample = torch.zeros(sample_num)
    for i in range(sample_num):
        y_pred = nn_model.predict(X[i])
        Y_pred[i] = y_pred
        loss_per_sample[i] = abs(y_pred-Y[i])
        if pprint:
            logmsg(" prediced Y:{}, Real Y:{} loss={}".format(loss_per_sample[i])) 
        
    total_loss = torch.norm((Y_pred-Y))/sample_num
    return total_loss,loss_per_sample.detach().numpy()

def calc_mre_loss(X,Y,nn_model,pprint=False):
    sample_num = X.shape[0] 
    print(str(sample_num))
    Y_pred = torch.zeros(sample_num)
    mre = torch.zeros(sample_num)
    for i in range(sample_num):
        y_pred = nn_model.predict(X[i])
        Y_pred[i] = y_pred
        mre[i] = abs((y_pred-Y[i])/Y[i])
        if pprint:
            logmsg(" prediced Y:{}, Real Y:{} loss={}".format(mre[i])) 
        
    total_loss = mre.sum()*(100/sample_num)
    return total_loss,mre.detach().numpy()

def plot_loss(test_loss,train_loss):
    ig, ax = plt.subplots(figsize=(12, 7))
    test_loss = test_loss.flatten()
    train_loss = train_loss.flatten()
    iterations = np.arange(train_loss.shape[0])
    ax.set_title("Training loss progress")
    ax.set_xlabel('Loss')
    ax.set_ylabel('Steps')
    ax.plot(iterations,
            test_loss,
            color="b",label="test loss")
    ax.plot(iterations,
            pow(train_loss,0.5),
            color="r",label="train_loss")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    
    # Settings
    epochs = 20
    train_set_size = 48 
    lr = 0.01

    # Process args from command shell
    argv = os.sys.argv[1:]
    args = parse_args(argv)
    if args.logfile:
        openlog(args.logfile)
    else:
        openlog()
    
    
    # Load data
    data_set_file = args.dataset
    #data_set_file = X264_mat.csv
    HCS_dataset = DataLoader(DATA_DIR,do_torch=True)
    HCS_dataset.load(data_set_file)

    
    # Init training and test data
    X = torch.zeros([train_set_size,HCS_dataset.num_features],dtype=torch.float32)
    Y = torch.zeros([train_set_size,1],dtype=torch.float32)
    X_test = torch.zeros([train_set_size,HCS_dataset.num_features],dtype=torch.float32)
    Y_test = torch.zeros([train_set_size,1],dtype=torch.float32)
    for i in range(train_set_size):
        X[i],Y[i]  = HCS_dataset[i]
        X_test[i],Y_test[i] = HCS_dataset[i+train_set_size]


    # Init neural net model
    nn = ExperimentalNN(num_features=HCS_dataset.num_features,
                        neuron_num=30,lr=lr)
    
    ## Train neural net with defined num of epochs, calc losses, plot data.
    train_loss = np.zeros([epochs,train_set_size])
    test_loss = np.zeros([epochs,train_set_size])
    #plot = False
    for epoch in range(epochs):    
        
        '''plot = False
        if i == 9:
            plot = True'''
        permutations = torch.randperm(train_set_size)
        shuffled_X = X[permutations]
        shuffled_Y = Y[permutations]
        train_loss[epoch] = nn.train_net(X=shuffled_X,Y=shuffled_Y,
                                         plot=False,save_train_data=True)
        epoch_total_loss,test_loss[epoch] = calc_mse_loss(X_test,Y_test,nn,pprint=False)
        logmsg("epoch test loss: {}".format(epoch_total_loss))


    final_test_mre_loss,_ = calc_mre_loss(X_test,Y_test,nn)
    print("\n\n#####\n# Test MRE Loss:{}".format(final_test_mre_loss))
    plot_loss(test_loss,train_loss)




    
    

    

