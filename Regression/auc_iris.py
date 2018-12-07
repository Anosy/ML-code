import numbers
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from numpy import interp
from sklearn import metrics
from itertools import cycle
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    np.random.seed(0)
    pd.set_option('display.width', 300)
    np.set_printoptions(suppress=True)
    data = pd.read_csv('iris.data', header=None)
    # le = LabelEncoder()
    # y = le.fit_transform(data[4])
    # print(y)
    iris_types = data[4].unique()    # series.unique()返回对象中的唯一值
    # print(iris_types)
    for i, iris_type in enumerate(iris_types):
        data.set_value(data[4] == iris_type, 4, i)  # 数值化标签
    x = data.iloc[:, :2]    #loc,iloc取行index,而iloc的特殊之处是只能取数值索引。且如果loc取[:2]表示取0，1，2行，而iloc取0，1
    n, features = x.shape
    y = data.iloc[:, -1].astype(np.int)
    c_number = np.unique(y).size  # 获取类别数量
    x, x_test, y, y_test = train_test_split(x, y, train_size=0.6, test_size=0.4, random_state=0)
    # print(y_test)
    y_one_hot = label_binarize(y_test, classes=np.arange(c_number))  # one_hot 编码,目的是匹配预测的prob或decision_function
    alpha = list(np.logspace(-2, 2, 20))  # 取alpha为10^-2到10^2中的10的t次方的20个值

    models = [
        ['KNN', KNeighborsClassifier(n_neighbors=7)],
        ['LogisticRegression', LogisticRegressionCV(Cs=alpha, penalty='l2', cv=3)],
        ['SVM(Linear)', GridSearchCV(SVC(kernel='linear', decision_function_shape='ovr'), param_grid={'C': alpha})],
        ['SVM(RBF)', GridSearchCV(SVC(kernel='rbf', decision_function_shape='ovr'), param_grid={'C': alpha, 'gamma': alpha})]
    ]

    colors = cycle('gmcr')
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(7, 6), facecolor='w')
    for (name, model), color in zip(models, colors):
        model.fit(x, y)
        if hasattr(model, 'C_'):  # hasattr(object, name)判断一个对象里面是否有name属性或者name方法，返回BOOL值，有name特性返回True， 否则返回False。
            print(name,model.C_)
        if hasattr(model, 'best_params_'):
            print(name,model.best_params_)
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(x_test)
        else:
            y_score = model.decision_function(x_test)
        '''sklearn.metrics.roc_curve  介绍 *ROC曲线只适合用在二分类问题上 
        Parameters:
            y_true : array, shape = [n_samples].作用：真实的二进制标志，如果标签是不是二进制，pos_label应明确给出。
            y_score : array, shape = [n_samples].作用：正例的预测值（得分）
            pos_label : int.作用：设定的值被认为是正面的标签，其他的被认为是负面的。
        Return:
            FPR：假正例率
            TPR: 真正例率    
            thresholds： 不断下降的权值来计算FPR,TPR
        '''
        fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())  # 计算每次取的阈值，以及随着改变的FPR和TPR
        auc = metrics.auc(fpr, tpr)
        print('AUC =',auc)
        plt.plot(fpr, tpr, c=color, lw=2, alpha=0.7, label=u'%s，AUC=%.3f' % (name, auc))
    plt.plot((0, 1), (0, 1), c='#808080', lw=2, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))  # 设置x轴刻度的表现方式,此处取(0, 1.1)每格为0.1长
    plt.yticks(np.arange(0, 1.1, 0.1))  # 设置y轴刻度的表现方式
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    # plt.legend(loc='lower right', fancybox=True, framealpha=0.8, edgecolor='#303030', fontsize=12)
    plt.title(u'鸢尾花数据不同分类器的ROC和AUC', fontsize=17)
    plt.show()

