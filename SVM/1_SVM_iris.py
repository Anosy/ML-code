import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'


if __name__ == "__main__":
    path = 'SVM\\iris.data'  # 数据文件路径
    data = pd.read_csv(path, header=None)  # 读取CSV（逗号分割）文件到DataFrame.  csv格式文件：每行相当于一条记录是用“，”分割字段的纯文本数据库文件
    cla = ['Iris-setosa','Iris-versicolor','Iris-virginica']
    cla = np.array(cla)
    x, y = data[list(range(4))], data[4]
    x = x[[0,1]]  # 选取前两个特征
    y = pd.Categorical(y).codes     # 效果如同进行了LabelEncoder，即 le = preprocessing.LabelEncoder() y = le.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=1,train_size=0.6, test_size=0.4)
    clf = svm.SVC(C=100, kernel='rbf', decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    print(clf.score(x_train, y_train.ravel()))
    print('训练集准确率:',accuracy_score(y_train,clf.predict(x_train)))
    print(clf.score(x_test, y_test.ravel()))
    print('测试集准确率:',accuracy_score(y_test,clf.predict(x_test)))

    # decision_function   其结果将返回3列的向量，且中每一行的数据表示图中点到三个分割面的距离，选择其中最大的作为类别
    print(type(clf.decision_function(x_train)))
    print('decision_function:\n',clf.decision_function(x_train))
    print('\npredict:\n',clf.predict(x_train))

    # 绘图
    x1_min, x2_min = x.min()
    x1_max, x2_max = x.max()
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # 从min到max分成500*500个点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    grid_hat = clf.predict(grid_test)       # 预测分类值
    grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
    # 解决matplotlib无法显示中文问题
    mpl.rcParams['font.sans-serif'] = [u'SimHei']       # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False          # 解决保存图像是负号'-'显示为方块的问题
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 对坐标轴进行着色
    plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)  # 样本
    # plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
    plt.xlabel(iris_feature[0], fontsize=13)  # x轴的坐标名
    plt.ylabel(iris_feature[1], fontsize=13)
    plt.xlim(x1_min, x1_max)    # x轴长度限制
    plt.ylim(x2_min, x2_max)  # y轴长度限制
    plt.title(u'鸢尾花SVM二特征分类', fontsize=16)  # 添加标题
    plt.grid(b=True, linestyle='-.')  #  显示网格
    plt.tight_layout(pad=1.5) # 紧凑显示图片，居中显示
    plt.show()









