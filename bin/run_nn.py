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
import math
from sklearn import preprocessing

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
    

def calc_epoch_mse_loss(X,Y,nn_model,pprint=False):
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
    return total_loss, loss_per_sample.detach().numpy()


def calc_epoch_mape_loss(X,Y,nn_model,pprint=False):
    print("\ncalc mre loss...")
    sample_num = X.shape[0] 
    print("samplenum:" + str(sample_num))
    Y_pred = torch.zeros(sample_num)
    mre_per_sample = torch.zeros(sample_num)
    for i in range(sample_num):
        y_pred = nn_model.predict(X[i])
        Y_pred[i] = y_pred
        mre_per_sample[i] = abs((y_pred-Y[i])/Y[i])*100
        if pprint:
            logmsg(" prediced Y:{}, Real Y:{} loss={}".format(y_pred,Y[i],mre_per_sample[i])) 
        
    total_mre_loss = mre_per_sample.sum()/sample_num
    return total_mre_loss,mre_per_sample.detach().numpy()

#TODO: add MRE loss to pytorch model and train on it.
'''def plot_mse_loss(test_loss,train_loss,xlabel,ylabel,
                  title="Training Loss Progress"):
    ig, ax = plt.subplots(figsize=(12, 7))
    test_loss = test_loss.flatten()
    train_loss = train_loss.flatten()
    iterations = np.arange(train_loss.shape[0])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(iterations,
            test_loss,
            color="b",label="test loss")
    ax.plot(iterations,
            pow(train_loss,0.5),
            color="r",label="train loss")
    ax.legend()
    plt.show()'''
'''def plot_mse_loss(test_loss,train_loss,xlabel,ylabel,
                  title="Training Loss Progress"):
    fig, ax = plt.subplots(figsize=(12, 7))
    iterations = np.arange(train_loss.shape[0])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(iterations,
            test_loss,
            color="b",label="test loss")
    ax.plot(iterations,
            pow(train_loss,0.5),
            color="r",label="train loss")
    ax.legend()
    plt.show()'''


def plot_mse_loss(test_loss,train_loss,xlabel,ylabel,
                  title="Training Loss Progress"):
    fig, ax = plt.subplots(figsize=(12, 7))
    iterations = np.arange(train_loss.shape[0])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(iterations,
            test_loss,
            color="b",label="test loss")
    ax.plot(iterations,
            train_loss,
            color="r",label="train loss")
    ax.legend()
    plt.show()

    
def run(epochs, train_set_size, test_set_size, lr, batch_size, 
        neuron_num,lamda):
  
    data_set_file = args.dataset
    #data_set_file = X264_mat.csv
    HCS_dataset = DataLoader(DATA_DIR,do_torch=True)
    HCS_dataset.load(data_set_file)
    
    # Init training and test data
    X = torch.zeros([train_set_size,HCS_dataset.num_features], 
                    dtype=torch.float64)
    X_test = torch.zeros([test_set_size,HCS_dataset.num_features],
                         dtype=torch.float64)
    Y = torch.zeros([train_set_size,1],dtype=torch.float64)
    Y_test = torch.zeros([test_set_size,1],dtype=torch.float64)
    for i in range(train_set_size):
        X[i],Y[i]  = HCS_dataset[i]
    for i in range(test_set_size):
            X_test[i],Y_test[i] = HCS_dataset[i+train_set_size]
    X=torch.as_tensor(preprocessing.scale(X),dtype=torch.float64)
    X_test=torch.as_tensor(preprocessing.scale(X_test),dtype=torch.float64)
    


    # Init neural net model
    nn = ExperimentalNN(num_features=HCS_dataset.num_features,
                        neuron_num=neuron_num, lr=lr, lamda=lamda)
    
    ## Train neural net with defined num of epochs, calc losses, plot data.
    epoch_step_num = int(math.ceil(train_set_size/batch_size))
    #train_loss = np.zeros([epochs, epoch_step_num])
    #test_loss = np.zeros([epochs, epoch_step_num])
    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    mse_loss_fn = torch.nn.MSELoss()

    logmsg("**")
    logmsg("** Starting run for HCS {} with:".split('.')[0])
    logmsg("** dataset samples num={}, trainset size={}, testset size={}," 
               " batch size={}, epochs={}".format(
               len(HCS_dataset), train_set_size, train_set_size, batch_size, epochs))
    logmsg("**\n")
    for epoch in range(epochs):    
        
        '''plot = False
        if i == 9:
            plot = True'''
        permutations = torch.randperm(train_set_size)
        shuffled_X = X[permutations]
        shuffled_Y = Y[permutations]
        train_loss[epoch]  = \
                     nn.train_net(X=shuffled_X, Y=shuffled_Y, plot=False, 
                                  save_train_data=False, batch_size=batch_size)
        #test_loss[epoch],_ = calc_epoch_mse_loss(X_test,Y_test,nn, pprint=False)
        Y_test_pred = nn.predict(X_test).detach()
        test_loss[epoch] = pow(mse_loss_fn(Y_test_pred,Y_test).detach().item(),0.5)
                           
        if epoch % 1 == 0:
            logmsg("epoch {} test loss: {}".format(epoch,test_loss[epoch]))

   
    final_test_mre_loss,_ = calc_epoch_mape_loss(X_test,Y_test,nn,pprint=False)
    print("\n\n#####\n# Test MRE Loss:{}".format(final_test_mre_loss))
    plot_mse_loss(test_loss, train_loss, ylabel="Root of MSE Loss",xlabel="Steps")


def configHMCS():
    epochs = 8000
    train_set_size = 77
    test_set_size = 400
    lr = 0.0003
    batch_size = 32
    num_neurons = 30
    lamda = 0
    '''
    epochs = 4000
    train_set_size = 77
    test_set_size = 400
    lr = 0.001
    batch_size = 8
    num_neurons = 30
    '''
    run(epochs=epochs, train_set_size=train_set_size,
        test_set_size=test_set_size, lr=lr, batch_size=batch_size,
        num_neurons=num_neurons,lamda=lamda)

def configX264():
    epochs = 500
    train_set_size = 32
    test_set_size = 100
    lr = 0.001
    batch_size = 16
    neuron_num = 200
    lamda = 1
    '''
    epochs = 4000
    train_set_size = 77
    test_set_size = 400
    lr = 0.001
    batch_size = 8
    num_neurons = 30
    '''
    run(epochs=epochs, train_set_size=train_set_size,
        test_set_size=test_set_size, lr=lr, batch_size=batch_size,
        neuron_num=neuron_num, lamda=lamda)

def configAxTLS():
    epochs = 1000
    train_set_size = 460
    test_set_size = 200
    lr = 0.00006
    batch_size = 115
    neuron_num = 600
    lamda = 0.01
    '''
    epochs = 4000
    train_set_size = 77
    test_set_size = 400
    lr = 0.001
    batch_size = 8
    num_neurons = 30
    '''
    run(epochs=epochs, train_set_size=train_set_size,
        test_set_size=test_set_size, lr=lr, batch_size=batch_size,
        neuron_num=neuron_num, lamda=lamda)


if __name__ == "__main__":
    
    # Settings
    '''epochs = 700
    train_set_size = 528
    test_set_size = 528
    lr = 0.00003
    batch_size = 140'''
    # Process args from command shell
    argv = os.sys.argv[1:]
    args = parse_args(argv)
    if args.logfile:
        openlog(args.logfile)
    else:
        openlog()

    configAxTLS()
    
    
    


    
    

    

