import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils import logmsg

## Simply Deep NN design: 
# Two hidden layers with RELU activation, and linear activation at the output
# zeros as intialization for the weights. No regularization.
class TrialNet(torch.nn.Module):
    def __init__(self, obs_size, hidden_size, hidden2_size, output_size):
        super(TrialNet, self).__init__()
        self.hidden1 = torch.nn.Linear(obs_size, hidden_size)
        self.hidden2 = torch.nn.Linear(hidden_size, hidden2_size)
        self.output = torch.nn.Linear(hidden2_size, output_size)
        self.hidden1.weight.data.fill_(0.00)
        self.hidden2.weight.data.fill_(0.00)
        self.output.weight.data.fill_(0.00)
        #print(self.hidden1.weight)
        #print(self.hidden2.weight)
        #print(self.output.weight)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.output(x)

class ExperimentalNN():

    def __init__(self, num_features, neuron_num, lr):
        
        self.input_dim = num_features
        self.net = TrialNet(self.input_dim,neuron_num,neuron_num,1)
        self.loss_fn = torch.nn.MSELoss()
        self.lr = lr

        # Adam optimizer with recommended paramenters (0.9,0.999)
        self.optimizer=torch.optim.Adam(self.net.parameters(), 
                                        lr=self.lr,betas=[0.9,0.999]) 

    def predict(self, x):
        return self.net(x)


    def train_step(self, x, y):

        # Make a prediction of the data using the network
        y_pred = self.net(x)

        # Calc loss
        loss = self.loss_fn(y_pred,y)
        logmsg(" prediced Y:{}, Real Y:{} loss={}".format(y_pred,y,pow(loss,0.5)))

        # Reset the Gradient Descent optimizer
        self.optimizer.zero_grad()

        # Back progogate and let the GD optimizer update the parameters(weights)
        loss.backward()
        self.optimizer.step()
        return loss,y_pred


    # Sample by sample training, TODO: add batch size
    def train_net(self, X, Y, save_train_data=True, plot=False):
        train_loss_for_step = torch.zeros(X.shape[0])
        Y_pred = torch.zeros(X.shape[0])
        for i,x in enumerate(X):
            loss_at_step,Y_pred[i] = self.train_step(x,Y[i])
            if save_train_data or plot:
                train_loss_for_step[i] = loss_at_step
        if plot:
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.set_title("Training loss progress")
            ax.set_xlabel('Loss')
            ax.set_ylabel('Steps')
            ax.plot(np.arange(X.shape[0]),pow(train_loss_for_step.detach().numpy(),0.5),
                    color="g",lw="2")
            plt.show()
     
        logmsg("------------------------------")
        logmsg("epoch train loss (root of MSE):{}"
               .format(torch.norm((Y_pred-Y),2)/X.shape[0]))
        
        if save_train_data:
            return train_loss_for_step.detach().numpy()
        else:
            return None



