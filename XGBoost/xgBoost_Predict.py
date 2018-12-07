import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    path = 'C:\\work_station\\xiaoxiang\\Regression\\iris.data'
    data = pd.read_csv(path,header=None)
    x, y = data[list(range(4))], data[4]
    y = pd.Categorical(y).codes
    x_train, x_test, y_trian, y_test = train_test_split(x, y, random_state=1, test_size=50)

    data_train = xgb.DMatrix(x_train, label=y_trian)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'test'), (data_train, 'train')]  # 设置观察表
    param = {'max_depth': 2, 'eta': 0.3, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}  # 设置参数
    num_round = 10

    # 训练模型
    bst = xgb.train(param, data_train, num_boost_round=num_round, evals=watch_list)
    y_hat = bst.predict(data_test)
    acc = float(sum(y_test==y_hat)) / len(y_hat)
    print('正确率为:\t',acc)
