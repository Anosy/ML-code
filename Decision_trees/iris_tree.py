import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pydotplus

# 花萼长度、花萼宽度，花瓣长度，花瓣宽度
iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = 'C:\\work_station\\xiaoxiang\Regression\\iris.data'
    data = pd.read_csv(path,header=None)
    x = data.iloc[:, :2]
    y = pd.Categorical(data[4]).codes
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=1)

    # 搭建模型
    model = DecisionTreeClassifier(criterion='entropy',max_depth=3)
    model.fit(x_train, y_train)
    y_test_hat = model.predict(x_test)  # 测试数据

    # 保存绘制的决策树文件
    dot_data = tree.export_graphviz(model, out_file=None, feature_names=iris_feature_E[:2], class_names=iris_class,
                                    filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('iris.pdf')

    # 画图
    x1_min, x2_min = x.min()
    x1_max, x2_max = x.max()
    t1 = np.linspace(x1_min, x1_max, 50)
    t2 = np.linspace(x2_min, x2_max, 50)
    x1, x2 = np.meshgrid(t1, t2)
    x_show = np.stack((x1.flat, x2.flat), axis=1)

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_show_hat = model.predict(x_show)  # 预测值
    y_show_hat = y_show_hat.reshape(x1.shape)  # 使其与输入的形状相同

    plt.figure(1)
    plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示
    plt.scatter(x_test[0], x_test[1], c=y_test.ravel(), edgecolors='k', s=150, zorder=10, cmap=cm_dark, marker='*')  # 测试数据
    plt.scatter(x[0], x[1], c=y.ravel(), edgecolors='k', s=40, cmap=cm_dark)  # 全部数据
    plt.xlabel(iris_feature[0], fontsize=15)
    plt.ylabel(iris_feature[1], fontsize=15)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(ls='-.')
    plt.title(u'鸢尾花数据的决策树分类', fontsize=17)
    plt.show()

    # 训练集上的预测结果
    count = 0
    y_test = y_test.reshape(-1)
    length = y_test.shape[0]
    for i in range(length):
        if y_test[i] == y_test_hat[i]:
            count += 1
    # print('准确度: %.2f%%' % (100 *(count/length)))
    #print(model.score(x_test,y_test))

    # 过拟合：错误率
    depth = np.arange(1, 15)
    err_list = []
    for d in depth:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
        clf.fit(x_train, y_train)
        acc = clf.score(x_test, y_test)
        err = 1 - acc
        err_list.append(err)
        print('层数为%d' % d, ' 错误率: %.2f%%' % (100 * err))
    plt.figure(2)
    plt.plot(depth, err_list, 'ro-', lw=2)
    plt.xlabel(u'决策树深度', fontsize=15)
    plt.ylabel(u'错误率', fontsize=15)
    plt.title(u'决策树深度与过拟合', fontsize=17)
    plt.grid(ls='-.')
    plt.show()
