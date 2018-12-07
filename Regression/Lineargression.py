import csv
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def linearRegression():
    path = 'Advertising.csv'
    # 手写读取数据
    # f = open(path)
    # x = []
    # y = []
    # for i, d in enumerate(f):
    #     if i == 0:
    #         continue
    #     d = d.strip()
    #     if not d:
    #         continue
    #     d = map(float,d.split(','))
    #     d = list(d)  # 注：在python3.x中如果希望输出list，则需要将map对象强制转化为list格式
    #     x.append(d[1:-1])
    #     y.append(d[-1])
    # x1 = np.array(x)
    # y = np.array(y)
    # print(x1)

    # python 自带库
    # x = []
    # y = []
    # f = open(path, 'r')
    # d = csv.reader(f)
    # for i, line in enumerate(d):
    #     # print(line)
    #     if i==0:
    #         continue
    #     if not line:
    #         continue
    #     for j,each in enumerate(line): # 通过系统默认读取的结果值都为字符串，需要强制转化
    #         line[j] = float(each)
    #     x.append(line[1:-1])
    #     y.append(line[-1])
    # x = np.array(x)
    # y = np.array(y)
    # print(x)
    # print(y)

    # numpy读入
    # x = []
    # y = []
    # p = np.loadtxt(path,delimiter=',', skiprows=1)  # delimiter用与分割参数，skiprows用于跳过第几行
    # for line in p:
    #     x.append(list(line[1:-1]))
    #     y.append(line[-1])
    # x = np.array(x)
    # y = np.array(y)
    # print(x)
    # print(y)

    # pandas读入
    data = pd.read_csv(path)
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    # print(x)
    # print(y)

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 绘图1
    fig = plt.figure()
    plt.plot(data['TV'],y,'ro',label='TV')
    plt.plot(data['Radio'], y, 'g^', label='Radio')
    plt.plot(data['Newspaper'], y, 'mv', label='Newspaper')
    plt.legend(loc='lower right')
    plt.xlabel(u'广告花费',fontsize=16)  # fontsize字体大小
    plt.ylabel(u'销售额',fontsize=16)
    plt.title(u'广告花费与销售额对比数据',fontsize=20)
    plt.grid(ls='-.')
    plt.show()

    # 绘图2
    fig = plt.figure(figsize=(9, 10))
    plt.subplot(311)
    plt.plot(data['TV'],y,'ro')
    plt.title('TV')
    plt.grid(ls='-.')
    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid(ls='-.')
    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'mv')
    plt.title('Newspaper')
    plt.grid(ls='-.')
    plt.tight_layout()
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=1)
    # print(type(x_test))
    # print(x_train.shape, y_train.shape)
    '''
    sklearn Regression介绍
    class sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    parameter：
        fit_intercept:boolean, optional, default True.作用：是否计算偏置b
        normalize : boolean, optional, default False.作用：将数值正规化
        copy_X : boolean, optional, default True.作用：复制x值
    attributes:
        coef_:array.属性：返回值为特征系数
        intercept_ : array.属性：返回值为截距
    '''
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    # print(model)
    # print(linreg.coef_,linreg.intercept_)

    order = y_test.argsort(axis=0)  # axis=0 表示对纵轴操作;axis=1 表示对纵轴操作
    y_test = y_test.values[order]
    # y_test = y_test.sort_values()  # sort_index()series通过索引进行排序，同理frame.sort_values(by='纵队名字')
    # print(y_test.values)
    x_test = x_test.values[order, :]  # 根据order的顺序来对x_test的中纵轴排序
    y_hat = linreg.predict(x_test)   # 利用linearRegression进行预测
    mse = np.average((y_hat - y_test) ** 2)  # 计算mse均方误差.np.average()计算加权平均值。
    rmse = np.sqrt(mse)  # 计算rmse均方根误差
    print('MSE =', mse)
    print('RMSE =', rmse)
    print('R2_train =', linreg.score(x_train, y_train))  # score方法返回的是R2值
    print('R2_test =', linreg.score(x_test, y_test))

    plt.figure()
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', lw=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', lw=2, label=u'预测数据')
    plt.legend(loc='upper right')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.grid(b=True, ls='-.')
    plt.show()


if __name__ == '__main__':
    linearRegression()


