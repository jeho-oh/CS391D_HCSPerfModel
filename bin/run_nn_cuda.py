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
import scipy.stats as stats

## Local modules from ../lib
from nn_experimental import ExperimentalNN, FFNet, TrialNetUniformInit
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
    

def calc_confidence(results):
     confidence = 0.95
     n = results.size
     print(n)
     std_err = stats.tstd(results)
     print(std_err)
     h = stats.norm.interval(0.95, loc=0, scale=std_err/math.sqrt(n))
     return h

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
    return total_loss, loss_per_sample.cpu().detach().numpy()


def calc_epoch_mape_loss(X,Y,nn_model,pprint=False):
    #print("\ncalc mre loss...")
    sample_num = X.shape[0] 
    #print("samplenum:" + str(sample_num))
    Y_pred = torch.zeros(sample_num)
    mre_per_sample = torch.zeros(sample_num)
    for i in range(sample_num):
        y_pred = nn_model.predict(X[i])
        Y_pred[i] = y_pred
        mre_per_sample[i] = abs((y_pred-Y[i])/Y[i])*100
        if pprint:
            logmsg(" prediced Y:{}, Real Y:{} loss={}".format(y_pred,Y[i],mre_per_sample[i])) 
        
    total_mre_loss = mre_per_sample.sum()/sample_num
    return total_mre_loss,mre_per_sample.cpu().detach().numpy()

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


def plot_loss(test_loss,train_loss,xlabel,ylabel,
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


def sortEigenVectors(evalues, evectors):
        evalues = evalues[:,0]
        non_zero_indices = torch.nonzero(torch.abs(evalues) > 1e-8, as_tuple=True)
        #print("evectors shape:" + str(evectors.size()))
        #print("indices:" + str(non_zero_indices))
        non_zero_evalues = evalues[non_zero_indices]
        #print("unsorted_evals " + str(non_zero_evalues))
        sorted_evalues,sorted_indices = torch.sort(non_zero_evalues,descending=True)
        #print("sorted evals " + str(sorted_evalues))
        #print("sorted indices: " + str(sorted_indices))
        #print("evectors shape: " + str(evectors[:,sorted_indices].size()))
        return sorted_evalues, evectors[:,sorted_indices]


def doPca(samples, eigenvec_num):
        K = samples.shape[0]
        #self.plotDigit(0)
        
        # Get K samples

        A_T = samples

        avg_X_T = torch.mean(A_T, axis=0)
        A_T -= avg_X_T

        # Find AA_T - which is the covariance matrix
        print(A_T.shape)
        A = A_T.t()

        cov_mat = torch.matmul(A,A_T)/K
        evalues, evectors = torch.eig(cov_mat, eigenvectors=True)
        #print(cov_mat.shape)
        #print("evectors:" + str(evectors.size()))
        #print("unsoreted evalues" + str(evalues))
        sorted_evalues,sorted_evectors = sortEigenVectors(evalues, evectors)
        #print("return back to doPca func...")
        #print("sorted evalues" + str(sorted_evalues.size()))
        #print("sorted evectors:" + str(evectors.size()))
        if (sorted_evectors.size()[1] > eigenvec_num):
            eigen_space_mat = sorted_evectors[:,0:eigenvec_num]
        else:
            eigen_space_mat = sorted_evectors
        #print("eigen mat size:" + str(eigen_space_mat.size()))
        return eigen_space_mat

        

    
def run(epochs, train_set_size, test_set_size, lr, batch_size, 
        neuron_num,lamda,plot=False,pca=False,eigenvec_num=30):
  
    data_set_file = args.dataset
    HCS_dataset = DataLoader(DATA_DIR,do_torch=True)
    HCS_dataset.load(data_set_file,scale=True,stack_features=0)
    #logmsg(HCS_dataset.num_features)
    #raise Exception()

    # Init training and test data
    X = torch.zeros([train_set_size,HCS_dataset.num_features], 
                    dtype=torch.float32)
    X_test = torch.zeros([test_set_size,HCS_dataset.num_features],
                         dtype=torch.float32)
    Y = torch.zeros([train_set_size,1],dtype=torch.float32)
    Y_test = torch.zeros([test_set_size,1],dtype=torch.float32)
    
    for i in range(train_set_size):
        X[i],Y[i]  = HCS_dataset[i]
    for i in range(test_set_size):
            X_test[i],Y_test[i] = HCS_dataset[i+train_set_size]
    

    feature_num = HCS_dataset.num_features
    if pca:
        eigen_transform_mat = doPca(X,eigenvec_num)
        X = torch.matmul(X,eigen_transform_mat)
        X_test = torch.matmul(X_test,eigen_transform_mat)
        feature_num = X.size()[1]
        print("pca feature num:" + feature_num)

    #X=torch.as_tensor(preprocessing.scale(X),dtype=torch.float64)
    #X_test=torch.as_tensor(preprocessing.scale(X_test),dtype=torch.float64)
    #Y = Y/100000
    #Y_test = Y_test/100000
    #X[:][X==0] = -1
    #X_test[:][X_test==0] = -1
    #print(X_test[1])
    #print(X[17])


    # Init neural net model
    '''depth = 8
    nn_topologhy = FFNet(feature_size=feature_num, depth=depth,
                         hidden_sizes=[neuron_num]*depth, output_size=1,
                         init="xav")'''
    nn = ExperimentalNN(num_features=feature_num,
                        neuron_num=neuron_num, lr=lr, lamda=lamda)
    '''nn = ModularDeepNN(num_features=feature_num,
                        nn_model = nn_topologhy, lr=lr, lamda=lamda)
    '''
    
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
        #train_loss[epoch],_ = calc_epoch_mape_loss(X,Y,nn, pprint=False,)
        #Y_test_pred = nn.predict(X_test).detach()
        test_loss[epoch],_ = calc_epoch_mape_loss(X_test,Y_test,nn)
                           
        if epoch % 20 == 0:
            logmsg("epoch {} test loss: {}".format(epoch,test_loss[epoch]))

   
    final_test_mre_loss,_ = calc_epoch_mape_loss(X_test,Y_test,nn,pprint=False)
    print("\n\n#####\n# Test MAPE Loss:{}".format(final_test_mre_loss))
    if plot:
        plot_loss(test_loss, train_loss, ylabel="MAPE",xlabel="Steps")
    return final_test_mre_loss


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

def configHipa():
    epochs = 500
    train_set_size = 528
    test_set_size = 528
    lr = 0.00015
    batch_size = 128
    neuron_num = 200
    lamda = 0.1
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
        neuron_num=neuron_num, lamda=lamda,pca=False, eigenvec_num=10)



def configAxTLS():
    epochs = 200
    test_set_size = 300
    lr = 0.0003
    batch_size = 800
    neuron_num = 300
    lamda = 0.01
    #torch.autograd.set_detect_anomaly(True)
    '''
    epochs = 4000
    train_set_size = 77
    test_set_size = 400
    lr = 0.001
    batch_size = 8
    num_neurons = 30
    '''
    experiments=5
    #18
    for train_set_size in [7500,15000]:
        batch_size = int(math.ceil(batch_size))
        results = np.zeros(experiments)
        for i in range(experiments):
            results[i] = run(epochs=epochs, train_set_size=train_set_size,
                test_set_size=test_set_size, lr=lr, batch_size=batch_size,
                neuron_num=neuron_num, lamda=lamda, plot=True, pca=False, eigenvec_num=1500)
            print("-------- Experiment {} Done  --------".format(i))
        print("    _________________________\n    finished SIZE {}".format(train_set_size))
        print(     "mean MRE:{}%   confidence interval:{} %".format(np.mean(results),calc_confidence(results)))
        print("    ------------------------\n\n")

def configFiasco():
    epochs = 500
    test_set_size = 300
    lr = 0.00005
    batch_size = 600
    neuron_num = 200
    lamda = 0.1
    #torch.autograd.set_detect_anomaly(True)
    '''
    epochs = 4000
    train_set_size = 77
    test_set_size = 400
    lr = 0.001
    batch_size = 8
    num_neurons = 30
    '''
    experiments=5
    for train_set_size in [1170]:
        batch_size = int(math.ceil(train_set_size/2))
        results = np.zeros(5)
        for i in range(experiments):
            results[i] = run(epochs=epochs, train_set_size=train_set_size,
                test_set_size=test_set_size, lr=lr, batch_size=batch_size,
                neuron_num=neuron_num, lamda=lamda, plot=True, pca=False, eigenvec_num=80)
            print("-------- Experiment Done --------")
        print("    _________________________\n    finished SIZE {}".format(train_set_size))
        print(     "mean MRE:{}%   confidence interval:{} %".format(np.mean(results),calc_confidence(results)))
        print("    ------------------------\n\n")

def configUCLib():
    epochs = 200
    train_set_size = 600
    test_set_size = 300
    lr = 0.0003
    batch_size = 600
    neuron_num = 200
    lamda = 0.1
    '''
    epochs = 4000
    train_set_size = 77
    test_set_size = 400
    lr = 0.001
    batch_size = 8
    num_neurons = 30
    '''
    #for train_set_size in [269,807,1345]:
    trials = 5
    for train_set_size in [269,807,1345]:
        batch_size = int(math.ceil(train_set_size/4))
        results = np.zeros(5)
        for i in range(trials):
            results[i] = run(epochs=epochs, train_set_size=train_set_size,
                test_set_size=test_set_size, lr=lr, batch_size=batch_size,
                neuron_num=neuron_num, lamda=lamda, pca=False, eigenvec_num=30, plot=True)
            print("-------- Experiment Done --------")
        print("    _________________________\n    finished SIZE {}".format(train_set_size))
        print(     "mean MRE:{}%   confidence interval:{} %".format(np.mean(results),calc_confidence(results)))
        print("    ------------------------\n\n")

if __name__ == "__main__":
    
    # Settings
    '''epochs = 700
    train_set_size = 528
    test_set_size = 528
    lr = 0.00003
    batch_size = 140'''
    # Process args from command shell
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    argv = os.sys.argv[1:]
    args = parse_args(argv)
    if args.logfile:
        openlog(args.logfile)
    else:
        openlog()
    
    #torch.manual_seed(0)
    #configHipa()
    #configAxTLS()
    #configFiasco()
    configUCLib()
    
    
    


    
    

    

