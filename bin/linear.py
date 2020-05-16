from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

import scipy.stats

import pandas as pd
import numpy as np
import os

# import warnings
# warnings.filterwarnings("ignore")

bindir = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(bindir, '..', 'Data', '')


def calc_mape_loss(Y, Y_pred):
    sample_num = Y.shape[0]
    mre_per_sample = np.zeros(sample_num)
    for i in range(sample_num):
        mre_per_sample[i] = abs((Y_pred[i] - Y[i]) / Y[i]) * 100

    total_mre_loss = mre_per_sample.sum() / sample_num
    return total_mre_loss


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


target = 'axtls_2_1_4'
reg = 'l1'
rep = 5

filepath = os.path.join(DATA_DIR, target + ".csv")
try:
    data_frame = pd.read_csv(filepath)
except Exception as e:
    raise Exception("Can't load data csv file. Error: {}".format(e))

num_samples = data_frame.shape[0]
num_features = data_frame.shape[1] - 1

X = np.zeros([num_samples, num_features])
Y = np.zeros([num_samples])
for index, row in data_frame.iterrows():
    X[index] = row.iloc[:-1].to_numpy()
    Y[index] = row.iloc[-1]

param_grid = [
  {'alpha': [1, 0.1, 0.01, 0.001], 'eta0': [0.1, 0.01, 0.001, 0.0001, 0.00001]},
 ]

for n in (250,):
    mape = np.zeros(rep)

    for k in range(rep):
        shuffled_indexes = np.random.permutation(np.arange(0, num_samples))
        X = X[shuffled_indexes]
        Y = Y[shuffled_indexes]

        train_size = 7500#n * num_features

        X_train = X[:train_size]
        Y_train = Y[:train_size]
        X_test = X[train_size: train_size + num_features]
        Y_test = Y[train_size: train_size + num_features]

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        est = SGDRegressor(max_iter=1000000, penalty=reg, learning_rate='constant')
        search = GridSearchCV(est, param_grid, cv=5, scoring='neg_mean_absolute_error', iid='True')
        search.fit(X_train, Y_train)
        #print(search.best_params_)

        Y_pred = search.best_estimator_.predict(X_test)
        mape[k] = calc_mape_loss(Y_test, Y_pred)

    avg = mape.sum() / rep
    print(str(mean_confidence_interval(mape, 0.95))[1:-1])
    # print(str(n) + "," + str(avg))
    # print(est.score(X_test, Y_test))

