import xgboost as xgb
import numpy as np


def log_reg(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0 - p)
    return g, h


def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)


if __name__ == '__main__':
    data_train = xgb.DMatrix('agaricus_train.txt')
    data_test = xgb.DMatrix('agaricus_test.txt')
    # print(data_train)
    # print(type(data_train))

    # 设置参数
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}  # 常规参数
    watchlist = [(data_test, 'test'), (data_train, 'train')]
    n_round = 7
    # bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    # 自定义目标函数和评价函数
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=log_reg, feval=error_rate)

    # 计算错误率
    y_hat = bst.predict(data_test)  # 测试集预测结果
    y = data_test.get_label()  # 测试集实际结果
    print(y_hat)
    print(y)

    error = sum(y != (y_hat > 0.5))
    error_rate = float(error) / len(y_hat)
    print('样本总数：    ', len(y_hat))
    print('错误数目:    %4d' % error)
    print('错误率：     %.5f%%' % error_rate)
