import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from enum import Enum
from utils import logmsg

## Simply Deep NN design: 
# Two hidden layers with RELU activation, and linear activation at the output
# zeros as intialization for the weights. No regularization.

def MAPELoss(output,target,pprint=False):
    #print("target:")
    #print(target)
    sample_num = output.size()[0] 
    
    mre = 100*torch.abs(output-target)/target
    mre = mre.sum()/sample_num
    #print("mre:")
    #print(mre)
    return mre

def MRE_MaxLoss(output,target,pprint=False):
    #print("target:")
    #print(target)
    sample_num = output.size()[0] 
    
    mre = 100*torch.abs(output-target)/target
    mre = torch.max(mre)
    #print("mre:")
    #print(mre)
    return mre.clone()


class TrialNetUniformInit(torch.nn.Module):
    def __init__(self, obs_size, hidden_size, hidden2_size, output_size):
        super(TrialNetUniformInit, self).__init__()
        # TODO: change layers to array
        # Add automation
        # Add regression on results for X264 to do this without pain
        self.hidden1 = torch.nn.Linear(obs_size, hidden_size)
        self.hidden2 = torch.nn.Linear(hidden_size, hidden2_size)
        self.hidden3 = torch.nn.Linear(hidden2_size, hidden2_size)
        self.hidden4 = torch.nn.Linear(hidden2_size, hidden2_size)
        self.hidden5 = torch.nn.Linear(hidden2_size, hidden2_size)
        self.hidden6 = torch.nn.Linear(hidden2_size, hidden2_size)
        self.hidden7 = torch.nn.Linear(hidden2_size, hidden2_size)
        self.hidden8 = torch.nn.Linear(hidden2_size, hidden2_size)
        self.output = torch.nn.Linear(hidden2_size, output_size)
        
        
        torch.nn.init.uniform_(self.hidden1.weight,b=0.00001)
        torch.nn.init.uniform_(self.hidden2.weight,b=0.00001)
        torch.nn.init.uniform_(self.hidden3.weight,b=0.00001)
        torch.nn.init.uniform_(self.hidden4.weight,b=0.00001)
        torch.nn.init.uniform_(self.hidden5.weight,b=0.00001)
        torch.nn.init.uniform_(self.hidden6.weight,b=0.00001)
        torch.nn.init.uniform_(self.hidden7.weight,b=0.00001)
        torch.nn.init.uniform_(self.hidden8.weight,b=0.00001)
        

        torch.nn.init.uniform_(self.output.weight,b=0.00001)

        #print(self.hidden1.weight)
        #print(self.hidden2.weight)
        #print(self.output.weight)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = F.relu(self.hidden7(x))
        x = F.relu(self.hidden8(x))
               
        return self.output(x)

    def get_layer_W(self):
        return self.hidden1.weight


class TrialNetNormInit(torch.nn.Module):
    def __init__(self, obs_size, hidden_size, hidden2_size, output_size):
        super(TrialNetNormInit, self).__init__()
        # TODO: change layers to array
        # Add automation
        # Add regression on results for X264 to do this without pain
        self.hidden1 = torch.nn.Linear(obs_size, hidden_size)
        self.hidden2 = torch.nn.Linear(hidden_size, hidden2_size)
        self.hidden3 = torch.nn.Linear(hidden2_size, hidden2_size)
        self.hidden4 = torch.nn.Linear(hidden2_size, hidden2_size)
        self.hidden5 = torch.nn.Linear(hidden2_size, hidden2_size)
        self.hidden6 = torch.nn.Linear(hidden2_size, hidden2_size)
        self.hidden7 = torch.nn.Linear(hidden2_size, hidden2_size)
        self.hidden8 = torch.nn.Linear(hidden2_size, hidden2_size)
                

        self.output = torch.nn.Linear(hidden2_size, output_size)
        torch.nn.init.normal_(self.hidden1.weight,mean=-0.1,std=0.3)
        torch.nn.init.normal_(self.hidden2.weight,mean=0.1,std=0.3)
        torch.nn.init.normal_(self.hidden3.weight,mean=-0.1,std=0.3)
        torch.nn.init.normal_(self.hidden4.weight,mean=0.1,std=0.3)
        torch.nn.init.normal_(self.hidden5.weight,mean=-0.1, std=0.3)
        torch.nn.init.normal_(self.hidden6.weight,mean=0.1,std=0.3)
        torch.nn.init.normal_(self.hidden7.weight,mean=-0.1,std=0.3)
        torch.nn.init.normal_(self.hidden8.weight,mean=0.1,std=0.3)
        

        torch.nn.init.normal_(self.output.weight,mean=-0.1,std=0.3)
        #print(self.hidden1.weight)
        #print(self.hidden2.weight)
        #print(self.output.weight)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = F.relu(self.hidden7(x))
        x = F.relu(self.hidden8(x))
               
        return self.output(x)

    def get_layer_W(self):
        return self.hidden1.weight

class TrialNetXavInit(torch.nn.Module):
    def __init__(self, obs_size, hidden_size, hidden2_size, output_size):
        super(TrialNetXavInit, self).__init__()
        # TODO: change layers to array
        # Add automation
        # Add regression on results for X264 to do this without pain
        self.hidden1 = torch.nn.Linear(obs_size, hidden_size)
        self.hidden2 = torch.nn.Linear(hidden_size, hidden2_size)
        self.hidden3 = torch.nn.Linear(hidden2_size, hidden2_size)
        self.hidden4 = torch.nn.Linear(hidden2_size, hidden2_size)
        self.hidden5 = torch.nn.Linear(hidden2_size, hidden2_size)
        self.hidden6 = torch.nn.Linear(hidden2_size, hidden2_size)
        self.hidden7 = torch.nn.Linear(hidden2_size, hidden2_size)
        self.hidden8 = torch.nn.Linear(hidden2_size, hidden2_size)
                
        self.output = torch.nn.Linear(hidden2_size, output_size)

        torch.nn.init.xavier_normal_(self.hidden1.weight,
                                    gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.hidden2.weight,gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.hidden3.weight,gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.hidden4.weight,gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.hidden5.weight,gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.hidden6.weight,gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.hidden7.weight,gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.hidden8.weight,gain=torch.nn.init.calculate_gain('relu'))
        
        torch.nn.init.xavier_normal_(self.output.weight,gain=torch.nn.init.calculate_gain('linear'))
        

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = F.relu(self.hidden7(x))
        x = F.relu(self.hidden8(x))
             
        return self.output(x)

    def get_layer_W(self):
        return self.hidden1.weight


class FFNet(torch.nn.Module):
    def __init__(self, feature_size, depth, hidden_sizes, output_size,
                 init="xav",init_std=0.1):
        super(FFNet, self).__init__()
        assert len(hidden_sizes) == depth, "provided number of layer neuron"\
               "sizes need to be equal to the net's depth"

        self.depth = depth
        self.layers = []
        self.layers.append(torch.nn.Linear(feature_size, hidden_sizes[0]))        
        for i in range(0,depth-1):
            self.layers.append(torch.nn.Linear(hidden_sizes[i],hidden_sizes[i+1]))

            
        self.output = torch.nn.Linear(hidden_sizes[-1], output_size)
        

        for layer in self.layers:
            if init == "zero":
                layer.weight.data.fill_(0.00)
            elif init == "norm":
                torch.nn.init.normal_(layer.weight,mean=0.00,std=init_std)
            else:
                torch.nn.init.xavier_normal_(
                                    layer.weight,
                                    gain=torch.nn.init.calculate_gain('relu'))
        if init == "zero":
            self.output.weight.data.fill_(0.00)
        elif init == "norm":
            torch.nn.init.normal_(self.output.weight, mean=0.00,std=init_std)
        else:
            torch.nn.init.xavier_normal_(
                                   self.output.weight,
                                    gain=torch.nn.init.calculate_gain('relu'))
        
        #print(self.layers[-1.weight)
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        
        return self.output(x)

    def get_layer_W(self,i):
        return self.layers[i-1].weight



class ModularDeepNN():
    #def __init__(self, num_features, neuron_num, lr, lamda=0):
    def __init__(self, num_features, nn_model, lr, lamda=0):
        
        self.input_dim = num_features
        #self.net = TrialNetXavInit(self.input_dim, neuron_num, neuron_num, 1)
        self.net= nn_model
        depth=8
        #self.net = FFNet(feature_size=num_features, depth=depth,
        #                 hidden_sizes=[neuron_num]*depth, output_size=1,
        #                 init="zero")
        #self.net.double()
        self.loss_fn = torch.nn.MSELoss()
        self.lr = lr
        self.lambd_a = lamda # This is intentionaly spelled incorrect.Otherwise it's python keyword. 

        # Adam optimizer with recommended paramenters (0.9,0.999)
        #self.optimizer=torch.optim.Adam(self.net.parameters(), 
        #                                 lr=self.lr, betas=[0.99,0.999]) 
        self.optimizer = torch.optim.SGD(self.net.parameters(),
                                         lr=self.lr)

    def predict(self, x):
        return self.net(x)

    ## Train net for a single mini-batch step
    def train_step(self, x, y):

        # Make a prediction of the data using the network
        y_pred = self.net(x)

        # Calc loss
        loss = self.loss_fn(y_pred,y)
        #logmsg(" predicted Y:{}, Real Y:{} loss={} avg diff={}".format(y_pred,y,pow(loss,0.5),(y_pred-y).abs().sum()/x.shape[0]))
        
        # Enable L1 (Lasso) Regularization for the first layer
        if self.lambd_a:
            #loss += self.lambd_a*torch.sum(torch.abs(self.net.get_layer_W(1)))
            loss += self.lambd_a*torch.sum(torch.abs(self.net.get_layer_W(1)))
            #loss += self.lambd_a/2*torch.sum(torch.abs(self.net.hidden2.weight))
            #loss += self.lambd_a*torch.sum(torch.abs(self.net.hidden8.weight))
            #loss += self.lambd_a*torch.sum(torch.abs(self.net.hidden4.weight))
         
        # Reset the Gradient Descent optimizer
        self.optimizer.zero_grad()

        # Back progogate and let the GD optimizer update the parameters(weights)
        loss.backward()
        self.optimizer.step()
        return loss, y_pred.squeeze()

    ## Train net for a single epoch
    def train_net(self, X, Y, X_test=None, Y_test=None,save_train_data=True,
                 calc_test_loss=False, plot=False, batch_size=8):
        samples_num = X.shape[0]
        steps_num = int(math.ceil(samples_num/batch_size))
        Y_pred = torch.zeros(samples_num)
        train_loss_for_step = torch.zeros(steps_num)
        test_loss_for_step = torch.zeros(steps_num) if calc_test_loss else None
        
        batch_indexes  = range(batch_size, samples_num, batch_size)
        batch_indexes  = list(batch_indexes) + [samples_num,]
        prev_batch_end = 0
        for i, batch_end in enumerate(batch_indexes):
            batch_start = prev_batch_end 
            loss_at_step,Y_pred[batch_start:batch_end] = \
                                                    self.train_step(X[batch_start:batch_end,:],
                                                        Y[batch_start:batch_end,:])
            prev_batch_end = batch_end
            if save_train_data or plot:
                train_loss_for_step[i] = self.loss_fn(Y_pred[batch_start:batch_end],
                                                  Y[batch_start:batch_end])
                #train_loss_for_step[i] = loss_at_step
                if calc_test_loss:
                    Y_pred_test = self.net(X_test)
                    test_loss_for_step[i] = self.loss_fn(Y_pred_test, Y_test) 
            if i+1 % 50 == 0:
                logmsg("Done {} training steps".format(i+1))

        Y_pred = Y_pred.unsqueeze(1)
        train_loss_for_step = train_loss_for_step.cpu().detach().numpy()
        if plot:
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.set_title("Training loss progress")
            ax.set_xlabel('Loss')
            ax.set_ylabel('Steps')
            ax.plot(np.arange(samples_num),pow(train_loss_for_step,0.5),
                    color="g",lw="2")
            plt.show()
     
        try_test_loss = ((Y_pred-Y).pow(2).sum()/samples_num).pow(0.5)
        #logmsg("epoch train loss (root of MSE):{} num_samples:{} Yshape: {} {}"
         #      .format(try_test_loss, samples_num, Y.shape, Y_pred.shape))
        
        
        if save_train_data:
            return train_loss_for_step, test_loss_for_step
        else:
            #print("output mse loss ++++++++++++++++++++++")
            Y_pred = self.net(X)
            #print("MSE loss fn out: " + str(pow(mse_loss_fn(Y_pred.squeeze(), Y.squeeze()).
            #                               detach().item(),0.5)) + " " + str(pow(mse_loss_fn(Y_pred, Y).
            #                               detach().item(),0.5)))
            return pow(mse_loss_fn(Y_pred, Y).detach().item(),0.5)

class ExperimentalNN():
    def __init__(self, num_features, neuron_num, lr, lamda=0):
        
        self.input_dim = num_features
        self.net = TrialNetXavInit(self.input_dim, neuron_num, neuron_num, 1)
        #self.net= nn_model
        
        #depth=8
        #self.net = FFNet(feature_size=num_features, depth=depth,
        #                 hidden_sizes=[neuron_num]*depth, output_size=1,
        #                 init="zero")
        #self.net.double()
        #self.loss_fn = torch.nn.MSELoss()
        #self.loss_fn = MRE_MaxLoss
        self.loss_fn = MAPELoss
        self.lr = lr
        self.lambd_a = lamda # This is intentionaly spelled incorrect.Otherwise it's python keyword. 

        # Adam optimizer with recommended paramenters (0.9,0.999)
        self.optimizer=torch.optim.Adam(self.net.parameters(), 
                          lr=self.lr, betas=[0.9,0.999],weight_decay=0.9) 
        #self.optimizer = torch.optim.SGD(self.net.parameters(),lr=self.lr)

    def predict(self, x):
        return self.net(x)

    ## Train net for a single mini-batch step
    def train_step(self, x, y):

        # Make a prediction of the data using the network
        y_pred = self.net(x)

        # Calc loss
        loss = self.loss_fn(y_pred,y)
        #logmsg(" predicted Y:{}, Real Y:{} loss={} avg diff={}".format(y_pred,y,pow(loss,0.5),(y_pred-y).abs().sum()/x.shape[0]))
        
        # Enable L1 (Lasso) Regularization for the first layer
        if self.lambd_a:
            #loss += self.lambd_a*torch.sum(torch.abs(self.net.get_layer_W(1)))
            loss += self.lambd_a*torch.sum(torch.abs(self.net.get_layer_W()))
            #loss += self.lambd_a/2*torch.sum(torch.abs(self.net.hidden2.weight))
            #loss += self.lambd_a*torch.sum(torch.abs(self.net.hidden8.weight))
            #loss += self.lambd_a*torch.sum(torch.abs(self.net.hidden4.weight))
         
        # Reset the Gradient Descent optimizer
        self.optimizer.zero_grad()

        # Back progogate and let the GD optimizer update the parameters(weights)
        loss.backward()
        self.optimizer.step()
        return loss, y_pred.squeeze()

    ## Train net for a single epoch
    def train_net(self, X, Y, X_test=None, Y_test=None,save_train_data=True,
                 calc_test_loss=False, plot=False, batch_size=8):
        samples_num = X.shape[0]
        steps_num = int(math.ceil(samples_num/batch_size))
        Y_pred = torch.zeros(samples_num)
        train_loss_for_step = torch.zeros(steps_num)
        test_loss_for_step = torch.zeros(steps_num) if calc_test_loss else None
        mse_loss_fn = torch.nn.MSELoss()
        
        batch_indexes  = range(batch_size, samples_num, batch_size)
        batch_indexes  = list(batch_indexes) + [samples_num,]
        prev_batch_end = 0
        for i, batch_end in enumerate(batch_indexes):
            batch_start = prev_batch_end 
            loss_at_step,Y_pred[batch_start:batch_end] = \
                                                    self.train_step(X[batch_start:batch_end,:],
                                                        Y[batch_start:batch_end,:])
            prev_batch_end = batch_end
            if save_train_data or plot:
                train_loss_for_step[i] = self.loss_fn(Y_pred[batch_start:batch_end],
                                                  Y[batch_start:batch_end])
                #train_loss_for_step[i] = loss_at_step
                if calc_test_loss:
                    Y_pred_test = self.net(X_test)
                    test_loss_for_step[i] = self.loss_fn(Y_pred_test, Y_test) 
            if i+1 % 50 == 0:
                logmsg("Done {} training steps".format(i+1))

        Y_pred = Y_pred.unsqueeze(1)
        train_loss_for_step = train_loss_for_step.cpu().detach().numpy()
        if plot:
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.set_title("Training loss progress")
            ax.set_xlabel('Loss')
            ax.set_ylabel('Steps')
            ax.plot(np.arange(samples_num),train_loss_for_step,
                    color="g",lw="2")
            plt.show()
     
        try_test_loss = ((Y_pred-Y).pow(2).sum()/samples_num).pow(0.5)
        #logmsg("epoch train loss (root of MSE):{} num_samples:{} Yshape: {} {}"
         #      .format(try_test_loss, samples_num, Y.shape, Y_pred.shape))
        
        
        if save_train_data:
            return train_loss_for_step, test_loss_for_step
        else:
            #print("output mse loss ++++++++++++++++++++++")
            Y_pred = self.net(X)
            #print("MSE loss fn out: " + str(pow(mse_loss_fn(Y_pred.squeeze(), Y.squeeze()).
            #                               detach().item(),0.5)) + " " + str(pow(mse_loss_fn(Y_pred, Y).
            #                               detach().item(),0.5)))
            return pow(mse_loss_fn(Y_pred, Y).detach().item(),0.5)



