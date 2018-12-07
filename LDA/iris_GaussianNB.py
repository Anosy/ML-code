import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]

def expand(a, b, rate=0.05):
    d = (b - a) * rate
    return a-d, b+d


if __name__ == '__main__':
    data_type = 'iris'

    if data_type=='car':
        colmun_names = 'buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability'
        data = pd.read_csv('car.data', header=None, names=colmun_names)
        for col in colmun_names:
            data[col] = pd.Categorical(data[col]).codes
        x = data[list(colmun_names[:-1])]
        y = data[colmun_names[-1]]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
        model = MultinomialNB(alpha=1)  # alpha=1 表示 Laplace平滑 alpha<1表示Lidstone
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        print('CAR训练集准确率：', accuracy_score(y_train, y_train_pred))
        y_test_pred = model.predict(x_test)
        print('CAR测试集准确率：', accuracy_score(y_test, y_test_pred))
    else:
        feature_names = u'花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度', '类别'
        data = pd.read_csv('..\\Regression\\iris.data', header=None, names=feature_names)
        x, y = data[list(feature_names[:-1])], data[feature_names[-1]]
        y = pd.Categorical(y).codes
        features = ['花萼长度', '花萼宽度']
        x = x[features]
        x, x_test, y, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        priors = np.array((1,2,4), dtype=float)
        priors /= priors.sum()
        gnb = Pipeline([
            ('sc', StandardScaler()),
            ('poly', PolynomialFeatures(degree=1)),
            ('clf', GaussianNB(priors=priors))])  # 由于鸢尾花数据是样本均衡的，其实不需要设置先验值
        gnb.fit(x, y.ravel())
        y_hat = gnb.predict(x)
        print('IRIS训练集准确度: %.2f%%' % (100 * accuracy_score(y, y_hat)))
        y_test_hat = gnb.predict(x_test)
        acc = 100 * accuracy_score(y_test, y_test_hat)
        print('IRIS测试集准确度：%.2f%%' % acc)
        acc = '准确率为：%.2f' % acc

        # 画图
        mpl.rcParams['font.sans-serif'] = ['simHei']
        mpl.rcParams['axes.unicode_minus'] = False
        cm_light = mpl.colors.ListedColormap(['#FF8080', '#77E0A0', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['r', 'g', '#6060FF'])
        x1_min, x2_min = x.min()
        x1_max, x2_max = x.max()
        x1_min, x1_max = expand(x1_min, x1_max)
        x2_min, x2_max = expand(x2_min, x2_max)
        x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
        grid_test = np.stack((x1.flat, x2.flat), axis=1)
        grid_hat = gnb.predict(grid_test)
        grid_hat = grid_hat.reshape(x1.shape)
        plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
        plt.scatter(x.ix[:,0], x.ix[:,1], s=20, c=y, marker='o', cmap=cm_dark, edgecolors='k')
        xx = 0.95 * x1_min + 0.05 * x1_max
        yy = 0.1 * x2_min + 0.9 * x2_max
        plt.text(xx, yy, acc, fontsize=15)
        plt.xlim((x1_min, x1_max))
        plt.ylim((x2_min, x2_max))
        plt.xlabel(features[0], fontsize=11)
        plt.ylabel(features[1], fontsize=11)
        plt.grid(b=True, ls=':', color='#606060')
        plt.title('高斯朴素贝叶斯分类鸢尾花数据', fontsize=14)
        plt.tight_layout(1, rect=(0, 0, 1, 0.95))
        plt.show()
