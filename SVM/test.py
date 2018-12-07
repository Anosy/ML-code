import csv
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    path = 'Advertising.csv'
    data = pd.read_csv(path)
    x = data[['TV', 'Radio']]
    y = data['Sales']

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    x_train, x_test, y_trian, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=1)
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_trian)

    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]
    y_hat = linreg.predict(x_test)
    mse = np.average((y_hat - y_test) ** 2)
    rmse = np.sqrt(mse)
    print('MSE =', mse)
    print('RMSE = ', rmse)
    print('R2_train = ', linreg.score(x_train, y_trian))
    print('R2_test =', linreg.score(x_test, y_test))

    plt.figure()
    t = np.arange(len(y_test))
    plt.plot(t, y_hat, 'r-', lw=2,label=u'预测的数据')
    plt.plot(t, y_test, 'g-', lw=2,label=u'真实的数据')
    plt.grid(True,ls='-.')
    plt.legend(loc='upper right')
    plt.title('线性回归预测销量',fontsize=18)
    plt.show()
