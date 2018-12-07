import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split   # cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd


if __name__ == '__main__':
    path = 'C:\\work_station\\xiaoxiang\XGBoost\\wine.data'
    data = pd.read_csv(path, header=None)
    x, y = data[list(range(1, 14))], data[0]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.5, test_size=0.5)

    # 逻辑回归
    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train)
    print('逻辑回归的准确率为%.4f' %  lr.score(x_test, y_test))

    # xgBoost
    y_train[y_train==3] = 0
    y_test[y_test==3] = 0
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'test'), (data_train, 'train')]
    params = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'multi:softmax', 'num_class':3}
    bst = xgb.train(params, data_train, num_boost_round=2, evals=watch_list)
    y_hat = bst.predict(data_test)
    acc = accuracy_score(y_test, y_hat)
    print('xgBoost的准确度为%.4f' % acc)
