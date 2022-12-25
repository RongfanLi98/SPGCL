# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    # data1 = np.mat(data.values) # no need
    if isinstance(data, np.matrix):
        data1 = data
    else:
        data1 = np.mat(data.values)
    train_size = int(time_len * rate)
    train_data = data1[0:train_size]
    test_data = data1[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])
    return trainX, trainY, testX, testY


data_set = 'HZY_west'
# mode = "real data"
mode = "normalize data"
path = r"..\datasets\HZY_west\HZY_west.csv"
data = np.loadtxt(fname=path, skiprows=1, delimiter=",").astype("float32")
data = data[:, 3:]
# data = pd.read_csv(path)
data = pd.DataFrame(data).transpose()
if mode == "real data":
    data = data
else:
    data_mean = data.mean()
    data_std = data.std()
    data = (data - data_mean) / data_std
    data = data.astype("float32")

data = data.fillna(0)
time_len = data.shape[0]
num_nodes = data.shape[1]
train_rate = 0.5
seq_len = 3
pre_len = 1
trainX, trainY, testX, testY = preprocess_data(data, time_len, train_rate, seq_len, pre_len)
method = 'ARIMA'  #### HA or SVR or ARIMA


def evaluation(labels, predicts, seq_len, pre_len, acc_threshold=0.05):
    """
    evalution the labels with predicts
    rmse, Root Mean Squared Error
    mae, mean_absolute_error
    F_norm, Frobenius norm
    Args:
        labels:
        predicts:
        acc_threshold: if lower than this threshold we regard it as accurate
    Returns:
    """
    labels = labels.reshape([-1, pre_len])
    predicts = predicts.reshape([-1, pre_len])
    labels = labels.squeeze(0).astype("float32")
    predicts = predicts.squeeze(0).astype("float32")
    rmse = mean_squared_error(y_true=labels, y_pred=predicts, squared=True)
    # mae = mean_squared_error(y_true=labels, y_pred=predicts, squared=False)
    mae = mean_absolute_error(y_true=np.array(labels), y_pred=np.array(predicts))
    r2 = r2_score(y_true=labels, y_pred=predicts)
    evs = explained_variance_score(y_true=labels, y_pred=predicts)
    # acc = a[np.abs(a - b) < np.abs(a * acc_threshold)]
    a, b = labels, predicts
    acc = a[np.abs(a - b) < np.abs(acc_threshold)]
    acc = np.size(acc) / np.size(a)
    # mape = MAPE(a, b)
    return rmse, mae, acc, r2, evs


########### HA #############

if method == 'HA':
    result = []
    for i in range(len(testX)):
        a = testX[i]
        a1 = np.mean(a, axis=0)
        result.append(a1)
    result1 = np.array(result)
    result1 = np.reshape(result1, [-1, num_nodes])
    testY1 = np.array(testY)
    testY1 = np.reshape(testY1, [-1, num_nodes])
    rmse, mae, accuracy, r2, var = evaluation(testY1, result1, seq_len, pre_len, acc_threshold=1)
    print('HA_rmse:%r' % rmse,
          'HA_mae:%r' % mae,
          'HA_acc:%r' % accuracy,
          'HA_r2:%r' % r2,
          'HA_var:%r' % var)
    rmse_ts, mae_ts, acc_ts, r2_ts, var_ts = rmse, mae, accuracy, r2, var

############ SVR #############

if method == 'SVR':
    total_rmse, total_mae, total_acc, result = [], [], [], []
    for i in range(num_nodes):
        data1 = np.mat(data.values)
        a = data1[:, i]
        a_X, a_Y, t_X, t_Y = preprocess_data(a, time_len, train_rate, seq_len, pre_len)
        a_X = np.array(a_X)
        a_X = np.reshape(a_X, [-1, seq_len])
        a_Y = np.array(a_Y)
        a_Y = np.reshape(a_Y, [-1, pre_len])
        a_Y = np.mean(a_Y, axis=1)
        t_X = np.array(t_X)
        t_X = np.reshape(t_X, [-1, seq_len])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y, [-1, pre_len])

        svr_model = SVR(kernel='linear')
        svr_model.fit(a_X, a_Y)
        pre = svr_model.predict(t_X)
        pre = np.array(np.transpose(np.mat(pre)))
        pre = pre.repeat(pre_len, axis=1)
        result.append(pre)
    result1 = np.array(result)
    result1 = np.reshape(result1, [num_nodes, -1])
    result1 = np.transpose(result1)
    testY1 = np.array(testY)

    testY1 = np.reshape(testY1, [-1, num_nodes])
    total = np.mat(total_acc)
    total[total < 0] = 0
    rmse, mae, accuracy, r2, var = evaluation(testY1, result1, seq_len, pre_len, acc_threshold=1)
    print('SVR_rmse:%r' % rmse,
          'SVR_mae:%r' % mae,
          'SVR_acc:%r' % accuracy,
          'SVR_r2:%r' % r2,
          'SVR_var:%r' % var)
    rmse_ts, mae_ts, acc_ts, r2_ts, var_ts = rmse, mae, accuracy, r2, var

######## ARIMA #########
if method == 'ARIMA':
    rmse, mae, acc, r2, var, pred, ori = [], [], [], [], [], [], []
    for i in range(200):
        ts = data.iloc[:, i]
        ts_log = np.log(ts)
        ts_log = np.array(ts_log, dtype=np.float)
        where_are_inf = np.isinf(ts_log)
        ts_log[where_are_inf] = 0
        where_are_nan = np.isnan(ts_log)
        ts_log[where_are_nan] = 0
        ts_log = pd.Series(ts_log)
        # ts_log.index = a1
        model = ARIMA(ts_log, order=[1, 0, 0])
        properModel = model.fit()
        predict_ts = properModel.predict(4, dynamic=True)
        log_recover = np.exp(predict_ts)
        ts = ts[log_recover.index]
        print(i)
        er_rmse, er_mae, er_acc, r2_score_s, var_score = evaluation(np.array(ts).reshape(-1).astype("float32"),
                                                                    np.array(log_recover).reshape(-1).astype("float32"),
                                                                    len(ts), len(ts), acc_threshold=1)
        rmse.append(er_rmse)
        mae.append(er_mae)
        acc.append(er_acc)
        r2.append(r2_score_s)
        var.append(var_score)
    acc1 = np.mat(acc)
    acc1[acc1 < 0] = 0
    print('arima_rmse:%r' % (np.mean(rmse)),
          'arima_mae:%r' % (np.mean(mae)),
          'arima_acc:%r' % (np.mean(acc1)),
          'arima_r2:%r' % (np.mean(r2)),
          'arima_var:%r' % (np.mean(var)))
    rmse_ts, mae_ts, acc_ts, r2_ts, var_ts = rmse, mae, acc1, r2, var

result_type = method
hyper_parameters = 'None'
time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

pure_data_file = r"..\pure_result.csv"
with open(pure_data_file, mode='a') as fin:
    str = "\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{},{}".format(rmse_ts, mae_ts, acc_ts, r2_ts, var_ts,
                                                                       time_stamp, hyper_parameters, data_set,
                                                                       result_type, mode)
    fin.write(str)
