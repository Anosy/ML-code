import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


if __name__ == '__main__':
    dataset = pd.read_csv('Social_Network_Ads.csv')
    x = dataset.iloc[:, [2,3]].values
    y = dataset.iloc[:, 4].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)

    d_train = lgb.Dataset(x_train, label=y_train)
    param = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt', # 选择GBDT作为基学习器
        'objective': 'binary', # 二分类
        'metric': 'binary_logloss', # 二分类问题，定义损失函数
        'sub_feature': 0.5, # feature_fraction 选择0.5的特征进行训练
        'min_data': 50, # min_data_in_leaf 一片叶子最小的数据量
        'max_depth': 10 # 最大深度为10

    }
    clf = lgb.train(param, d_train, num_boost_round=1000)
    y_pred= clf.predict(x_test)

    for i in range(x_test.shape[0]):
        if y_pred[i] > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cm)
    print(acc)

    # 防止过拟合的几个参数num_leaves, min_data_in_leaf, max_depth

    # 更快的速度bagging_fraction样本子重新， bagging_freq， feature_fraction 特征子采样，max_bin 选择较小的，save_binary
    # 加速未来数据的加载

    # 更高的精确度，使用较大的max_bin，但是可能造成运行速度慢；使用较小的学习率learning_rate,或者较大的迭代次数，使用大的叶子数量
    # 但是可能会产生过拟合；尝试使用dart

    # 解决过拟合问题，使用小的max_bin，使用小的num_leaves, 使用Use min_data_in_leaf 和 min_sum_hessian_in_leaf
    # 使用bagging_fraction and bagging_freq， 使用feature_fraction样本子采样，使用lambda_l1, lambda_l2，min_gain_to_split
    # 控制max_depth
