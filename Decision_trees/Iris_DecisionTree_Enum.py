import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 黑体 FangSong/KaiTi
    mpl.rcParams['axes.unicode_minus'] = False

    path = 'C:\\work_station\\xiaoxiang\\Regression\\iris.data'  # 数据文件路径
    data = pd.read_csv(path, header=None)
    x_prime = data[list(range(4))]
    y = pd.Categorical(data[4]).codes

    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    plt.figure(figsize=(10, 9), facecolor='#FFFFFF')
    for i, pair in enumerate(feature_pairs):
        x = x_prime[pair]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=1)
        clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
        clf.fit(x_train, y_train)
        print('特征：', iris_feature[pair[0]], ' + ', iris_feature[pair[1]])
        print('准确率为%.2f%%' % (clf.score(x_test, y_test) * 100))

        N, M = 500, 500  # 横纵各采样多少个值
        x1_min, x2_min = x.min()
        x1_max, x2_max = x.max()
        t1 = np.linspace(x1_min, x1_max, N)
        t2 = np.linspace(x2_min, x2_max, M)
        x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
        x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
        y_show_hat = clf.predict(x_show)  # 预测值
        y_show_hat = y_show_hat.reshape(x1.shape)  # 使其与输入的形状相同


        plt.subplot(2, 3, i+1)
        plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示
        plt.scatter(x_test[pair[0]], x_test[pair[1]], c=y_test.ravel(), edgecolors='k', s=150, zorder=10, cmap=cm_dark,
                    marker='*')  # 测试数据
        plt.scatter(x[pair[0]], x[pair[1]], c=y.ravel(), edgecolors='k', s=40, cmap=cm_dark)  # 全部数据
        plt.xlabel(iris_feature[pair[0]], fontsize=15)
        plt.ylabel(iris_feature[pair[1]], fontsize=15)
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.grid(ls='-.')
    plt.suptitle(u'决策树对鸢尾花数据的两特征组合的分类结果', fontsize=17)
    plt.tight_layout(2)
    plt.subplots_adjust(top=0.92)
    plt.show()
