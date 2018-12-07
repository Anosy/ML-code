import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    path = 'iris.data'  # 数据文件路径

    data = pd.read_csv('iris.data', header=None)
    data[4] = pd.Categorical(data[4]).codes
    x, y = np.split(data.values, (4,), axis=1)
    y = y.ravel()
    pca = PCA(n_components=2)
    x = pca.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0)

    lr = Pipeline([('sc', StandardScaler()),
                   ('poly', PolynomialFeatures(degree=2)),
                   ('clf', LogisticRegression())])
    lr.fit(x_train, y_train)  # y.ravel()将y形成一个扁平的数组
    y_hat = lr.predict(x_test)
    y_hat_prob = lr.predict_proba(x)
    print('y_hat = \n', y_hat)
    # print('y_hat_prob = \n', y_hat_prob)
    # print('训练准确度：%.2f%%' % (100*(np.mean(y_hat==y.ravel()))))
    print(u'训练准确度：%.2f%%' % (100 * lr.score(x_train, y_train)))
    print(u'测试准确度：%.2f%%' % (100 * lr.score(x_test, y_test)))
    # 画图
    N, M = 500, 500  # 横纵各采样多少个值
    x1_min, x1_max = x_test[:, 0].min(), x_test[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x_test[:, 1].min(), x_test[:, 1].max()  # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)

    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_test1 = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_hat = lr.predict(x_test1)  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)  # 预测值的显示
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, edgecolors='k', s=50, cmap=cm_dark)  # 样本的显示
    plt.xlabel(u'花萼长度', fontsize=14)
    plt.ylabel(u'花萼宽度', fontsize=14)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(ls='-.')
    patchs = [mpatches.Patch(color='#77E0A0', label='Iris-setosa'),
              mpatches.Patch(color='#FF8080', label='Iris-versicolor'),
              mpatches.Patch(color='#A0A0FF', label='Iris-virginica')]
    plt.legend(handles=patchs, fancybox=True, framealpha=0.8)
    plt.title(u'鸢尾花Logistic回归分类效果 - 标准化', fontsize=17)
    plt.show()
