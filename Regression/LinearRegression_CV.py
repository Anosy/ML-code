import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV



if __name__ == '__main__':
    data = pd.read_csv('Advertising.csv')
    # print(data)
    x = data[['TV','Radio','Newspaper']]
    y = data['Sales']

    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1) # 默认为3比1,ramdom_state设置为非零为了保持每次运行train，test固定
    # model = Lasso()
    model = Ridge()
    alpha_can = np.logspace(-3,2,num=10,base=10.0)
    # print(alpha_can)
    np.set_printoptions(suppress=True) # suppress消除小的数字使用科学记数法
    # print('alpha_can =',alpha_can)
    '''sklearn Lasso 介绍
    class sklearn.linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, 
        copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection=’cyclic’)
    Parameters:
        alpha : float, optional, defaults=1.0.作用：L1范数前的系数，当alpha=0等同与线性回归 ，alpha越大稀疏度越大
        fit_intercept : boolean  .作用：截距，默认为True
    Attributes：
        coef_：array.属性：特征系数
        intercept_ .属性：偏置
        
    '''
    '''sklearn GridSearchCV-网格搜索法（对估计值指定参数值的穷举搜索） 介绍
    class sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None, fit_params=None, n_jobs=1, iid=True,
     refit=True, cv=None, verbose=0, pre_dispatch=‘2*n_jobs’, error_score=’raise’, return_train_score=’warn’)
    Parameters:
        estimator : estimator object.作用：存放需要调参的模型
        param_grid : dict or list of dictionaries.作用：存放调制的参数.其中可以使用字典格式,字典的key为参数名,字典的
                values为参数值
        cv : int, cross-validation generator or an iterable, optional.作用：确定交叉验证分割策略。当cv=None默认使用3折，
                当cv=5表示使用5折交叉验证
        refit :默认为True,程序将会以交叉验证训练集得到的最佳参数，重新对所有可用的训练集与开发集进行，
                作为最终用于性能评估的最佳模型参数。即在搜索参数结束后，用最佳参数结果再次fit一遍全部数据集。        
    Attribution:
        cv_result : dict of numpy (masked) ndarrays
        grid_scores_：给出不同参数情况下的评价结果
        best_params_：描述了已取得最佳结果的参数的组合
        best_score_：成员提供优化过程期间观察到的最好的评分
    '''
    lasso_model = GridSearchCV(model, param_grid={'alpha':alpha_can}, cv=5)  # 使用网格搜索，选择最优的超参数
    lasso_model.fit(x_train, y_train)
    print('超参数:\n',lasso_model.best_params_)

    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]
    y_hat = lasso_model.predict(x_test)
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print('MSE =',mse)
    print('RMSE =',rmse)
    print('R2_train',lasso_model.score(x_train,y_train))
    print('R2_test',lasso_model.score(x_test,y_test))

    t = np.arange(len(x_test))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()


