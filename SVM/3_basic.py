import operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time
import math
import pandas as pd
import matplotlib.patches as mpatches
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV


def is_prime(x):
    return 0 not in [x % i for i in range(2, int(math.sqrt(x)) + 1)]


def extend(a, b):
    return 1.05 * a - 0.05 * b, 1.05 * b - 0.05 * a


def is_prime3(x):
    flag = True
    for p in p_list2:
        if p > math.sqrt(x):
            break
        if x % p == 0:
            flag = False
            break
    if flag:
        p_list2.append(x)
    return flag


def pca_function():
    pd.set_option('display.width', 200)
    data = pd.read_csv('iris.data', header=None)
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
    data.rename(columns=dict(zip(range(5), columns)),
                inplace=True)  # data.rename中，columns为key=旧，value=新的字典格式，且inplace=True才进行重命名
    data['type'] = pd.Categorical(data['type']).codes  # 将类别进行大小分类，如何通过codes将其转化为数值0123..
    # print(data)
    x = data.loc[:, columns[:-1]]  # type之外的列(所有行)
    y = data['type']

    # n_components：这个参数可以帮我们指定希望PCA降维后的特征维度数目
    # whiten ：判断是否进行白化。所谓白化，就是对降维后的数据的每个特征进行归一化，让方差都为1
    pca = PCA(n_components=2, whiten=True, random_state=0)
    x_pca = pca.fit_transform(x)
    # print('各方向方差：', pca.explained_variance_)
    # print('方差所占比例：', pca.explained_variance_ratio_)
    # print(x[:5])
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.scatter(x_pca[:, 0], x_pca[:, 1], s=30, c=y, marker='o', cmap=cm_dark)
    ax1.grid(True, ls='-.')
    ax2 = fig.add_subplot(212)
    ax2.scatter(x['sepal_length'], x['petal_length'], s=30, c=y, marker='o', cmap=cm_light)
    ax2.grid(True, ls='-.')
    plt.show()


if __name__ == '__main__':
    # 计算素数
    # a = 2
    # b = 100000

    # 方法1：直接计算
    # t = time()
    # p = [p for p in range(a, b) if 0 not in [p % d for d in range(2, int(math.sqrt(p)) + 1)]]
    # print(time()-t)
    # print(p)

    # 方法2: 利用filter
    # t = time()
    # p = list(filter(is_prime, range(a, b)))  # 在python3中，使用filter需要加上list转化
    # print(time() - t)
    # print(p)

    # 方法3：利用filter和lambda
    # t = time()
    # is_prime2 = (lambda x:0 not in [x % i for i in range(2, int(math.sqrt(x)) + 1)])
    # p = list(filter(is_prime2,range(a,b)))
    # print(time()-t)
    # print(p)

    # 方法4：定义
    # t = time()
    # p_list = []
    # for i in range(a,b):
    #     flag = True
    #     for p in p_list:
    #         if p > math.sqrt(i):
    #             break
    #         if i % p == 0:
    #             flag = False
    #             break
    #     if flag:
    #         p_list.append(i)
    # print(time() - t)
    # print(p_list)

    # 方法5
    # p_list2 = []
    # t = time()
    # list(filter(is_prime3, range(2, b)))
    # print(time()-t)
    # print(p_list2)

    # pandas
    # pd.set_option('display.width', 200) # 设置横向最多显示多少个字符， 一般80不适合横向的屏幕，平时多用200.
    # data = pd.read_excel('sales.xlsx', sheetname='sheet1', header=0) # header表示去掉抬头
    # print('data.head() = \n', data.head())
    # print('data.tail() = \n', data.tail())
    # print('data.dtypes = \n', data.dtypes)
    # print('data.columns = \n', data.columns)
    # for c in data.columns:
    #     print(c)
    # data['total'] = data['Jan'] + data['Feb'] + data['Mar']
    # print(data.head())
    # print(data['Jan'].sum())
    # print(data['Jan'].max())
    # print(data['Jan'].min())
    # print(data['Jan'].mean())
    # 添加一行
    # s1 = data[['Jan', 'Feb', 'Mar', 'total']].sum()
    # # print(s1)
    # s2 = pd.DataFrame(data=s1)
    # print(s2)
    # print(s2.T)
    # print(s2.T.reindex(columns=data.columns))
    # s = pd.DataFrame(data=data[['Jan','Feb', 'Mar', 'total']].sum()).T
    # s = s.reindex(columns=data.columns,fill_value=None)
    # # print(s)
    # data = data.append(s,ignore_index=True)
    # data = data.rename(index={15: 'Total'})
    # print(data.tail())

    pca_function()
