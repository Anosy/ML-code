import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import warnings


# import exceptions


def not_empty(s):
    return s != ''


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')  # 忽略所有的警告
    np.set_printoptions(suppress=True)
    file_data = pd.read_csv('housing.data', header=None)
    # print(file_data)
    data = np.empty([len(file_data), 14])
    for i, d in enumerate(file_data.values):
        d = list(map(float, filter(not_empty, d[0].split(' '))))
        data[i] = d
    x, y = np.split(data, (13,), axis=1)
    print('样本个数：%d, 特征个数：%d' % x.shape)
    # print(y.shape)
    y = y.ravel()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0)

    ''' sklearn ElasticNetCV 介绍（介于Lasso和Ridge回归的回归方法）
        class sklearn.linear_model.ElasticNetCV(l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, 
        normalize=False, precompute=’auto’, max_iter=1000, tol=0.0001, cv=None, copy_X=True, verbose=0, n_jobs=1,
         positive=False, random_state=None, selection=’cyclic’)
        目标函数：1 / (2 * n_samples) * ||y - Xw||^2_2+ alpha * l1_ratio * ||w||_1+ 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
        因此需要调参的超参数为alpha和l1_ratio
        Parameters:
            l1_ratio : float or array of floats, optional.作用：如果l1_ratio=0则为l2范数,如果l2_ratio=1则为l1范数
            alphas : numpy array, optional.
    '''
    model = Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures(degree=3, include_bias=True)),
        ('linear', ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.99, 1], alphas=np.logspace(-3, 2, 5),
                                fit_intercept=False, max_iter=1e3, cv=3))
    ])
    model.fit(x_train, y_train)
    linear = model.get_params('linear')['linear']
    # print(u'超参数：', linear.alpha_)
    print(u'L1 ratio：', linear.l1_ratio_)
    print(u'系数：', linear.coef_.ravel())

    order = y_test.argsort(axis=0)
    y_test = y_test[order]
    x_test = x_test[order, :]
    y_pred = model.predict(x_test)
    r2 = model.score(x_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print('R2:', r2)
    print(u'均方误差：', mse)

    t = np.arange(len(y_pred))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', lw=2, label=u'真实值')
    plt.plot(t, y_pred, 'g-', lw=2, label=u'估计值')
    plt.legend(loc='best')
    plt.title(u'波士顿房价预测', fontsize=18)
    plt.xlabel(u'样本编号', fontsize=15)
    plt.ylabel(u'房屋价格', fontsize=15)
    plt.grid(ls='dotted')
    plt.show()
