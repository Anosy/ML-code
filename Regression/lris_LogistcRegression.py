import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA



if __name__ == '__main__':
    path = 'iris.data'  # 数据文件路径

    # # 手动读取数据
    # f = open(path)
    # x = []
    # y = []
    # for line in f:
    #     line = line.strip()
    #     if line:
    #         line = line.split(',')
    #         x.append(list(map(float,line[:-1])))  # 使用map，将list中的每一个元素转化为float格式
    #         y.append(line[-1])
    # # print('原始数据X：\n', x)
    # # print('原始数据Y：\n', y)
    # x = np.array(x)
    # y = np.array(y)
    # print( 'Numpy格式X：\n', x)
    # # y[y == 'Iris-setosa'] = 0  # 手动编码
    # # y[y == 'Iris-versicolor'] = 1
    # # y[y == 'Iris-virginica'] = 2
    # # y = y.astype(dtype=np.int)
    # le = preprocessing.LabelEncoder()
    # y = le.fit_transform(y)
    # print( 'Numpy格式Y：\n',y)

    # # 使用sklearn的数据预处理
    # df = pd.read_csv(path, header=None)
    # x = df.values[:, :-1]  # 取所有行，除倒数第一列
    # y = df.values[:, -1]   # 取所有行，倒数第一列
    # '''sklearn LabelEncoder 介绍
    #     Attributes:
    #         classes_ : array of shape (n_class,).属性：持有的每个类的标签。
    #     Methods：
    #         fit(y)	Fit label encoder
    #         fit_transform(y)	Fit label encoder and return encoded labels
    #         inverse_transform(y)	Transform labels back to original encoding.
    #         transform(y)	Transform labels to normalized encoding.
    # '''
    # le = preprocessing.LabelEncoder() # 对标签值进行数值编码
    # le.fit(['Iris-setosa','Iris-versicolor','Iris-virginica'])
    # # print(le.classes_)
    # y = le.transform(y)
    # print('通过标签编码的Y：\n',y)

    data = pd.read_csv('iris.data', header=None)
    data[4] = pd.Categorical(data[4]).codes
    x, y = np.split(data.values, (4,), axis=1)
    # 使用PCA进行主成分分析，获取两个特征
    pca = PCA(n_components=2, whiten=True, random_state=0)
    x = pca.fit_transform(x)

    # 仅选择前两个特征
    # x = x[:, :2]
    '''sklearn LogisticRegression 介绍
        class sklearn.linear_model.LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
        intercept_scaling=1, class_weight=None, random_state=None, solver=’liblinear’, max_iter=100, multi_class=’ovr’,
         verbose=0, warm_start=False, n_jobs=1)
        Parameters:
            penalty : str, ‘l1’ or ‘l2’, default: ‘l2’
            class_weight：用于标示分类模型中各种类型的权重，可以是一个字典或者balanced字符串，默认为不输入，也就是不考虑权重，即为None。
            作用：1.如果误分类代价很高，那么就需要提高不希望被误分类的类的权重，保证其较大程度不被误分类
                  2.如果样本存在高度的失衡性，也就是说可能一个类的样本很多，但是另外一个类的样本较少，这时就需要提高少样本的权重，保持其在训练中的地位
            solver：优化算法选择参数，只有五个可选参数，即newton-cg,lbfgs,liblinear,sag,saga。默认为liblinear。
                liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
                lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
                newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
                sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。
                saga：线性收敛的随机优化算法的的变重。
                一般如果样本数量较小的时候选择liblinear就够了，但是如果样本较大，可以考虑使用sag和saga。
            multi_class：分类方式选择参数，str类型，可选参数为ovr和multinomial，默认为ovr。如果选择了ovr那么损失函数的优化方法就可以选择全部，
                但是如果选择为multinomial那么就只能选newton-cg, lbfgs和sag
    '''
    ''' sklearn StandardScaler 介绍（Standardization即标准化，尽量将数据转化为均值为零，方差为一的数据，形如标准正态分布）
        sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
        目的：解决了某些特征的区间过大，而一些区间过小，即假设x[0]变化的区间在1-10，X[1]变化的区间在20-200，这样y=X[0]W[0]+X[1]W[1]中，
        只要W[1]稍微改变一点就对y的值影响很大，而W[0]的值变化一些影响相对较小。
    '''
    ''' sklearn PolynomialFeatures 介绍   生成一个新的特征矩阵，该矩阵由具有小于或等于指定度的特征的所有多项式组合组成。
        class sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
        if degree=2,then [a, b]---->[1, a, b, a^2, ab, b^2]
        Parameters:
            degree : integer.The degree of the polynomial features. Default = 2.
            
    '''
    '''sklearn Pipeline 介绍 (一般结合上交叉验证，使得步骤变得简单)
        class sklearn.pipeline.Pipeline(steps, memory=None)
        *pipeline可以用于把多个estimators级联成一个estimator，这么做的原因是考虑了数据处理过程中一系列前后相继的固定流程*
        注意：Pipleline中最后一个之外的所有estimators都必须是变换器（transformers），最后一个estimator可以是任意类型（transformer，classifier，regresser）
        Parameters:	
            steps : list.Pipeline对象接受二元tuple构成的list，每一个二元 tuple 中的第一个元素为 arbitrary identifier string,
                    二元 tuple 中的第二个元素是 scikit-learn与之相适配的transformer 或者 estimator。
    '''
    lr = Pipeline([('sc', StandardScaler()),
                   ('poly', PolynomialFeatures(degree=4)),
                   ('clf', LogisticRegression())])
    lr.fit(x, y.ravel())  # y.ravel()将y形成一个扁平的数组
    y_hat = lr.predict(x)
    y_hat_prob = lr.predict_proba(x)
    # print('y_hat = \n', y_hat)
    # print('y_hat_prob = \n', y_hat_prob)
    # print('训练准确度：%.2f%%' % (100*(np.mean(y_hat==y.ravel()))))
    print(u'训练准确度：%.2f%%' % (100*lr.score(x,y)))
    # 画图
    N, M = 500, 500  # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    """example:
    t1 = np.arange(1, 4)
    t2 = np.arange(1, 5)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    print(x_test)
    相当于输出坐标点(1,1)(2,1)(3,1)(1,2)(2,2)(3,2)(1,3)(2,3)(3,3)(1,4)(2,4)(3,4)(4,4)
    """
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_hat = lr.predict(x_test)  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)  # 预测值的显示
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)  # 样本的显示
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
