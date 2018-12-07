import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
from scipy.optimize import leastsq
from scipy import stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
from scipy.interpolate import BarycentricInterpolator
from scipy.interpolate import CubicSpline
import math

if __name__ == '__main__':
    # L = [1, 2, 3, 4, 5, 6]
    # print('L =',L)
    # a = np.array(L)
    # print('a =',a)
    # d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float)
    # print(d)
    # e = d.astype(np.int)
    # print(e)
    # f = e.reshape(4,-1)
    # print(f)

    # print(np.arange(10,20))
    # print(np.linspace(1,10,10))
    # print(np.logspace(1, 2, 2, base=2))

    # a = np.arange(10)
    # print('a =',a)
    # print(a[::2])
    # print(a[::-1])

    # a = np.logspace(0, 9, 10, base=2)
    # print(a)
    # i = np.arange(0, 10, 2)
    # print(i)
    # b = a[i]
    # print(b)

    # a = np.random.rand(10)
    # print(a)
    # print(a>0.5)
    # b = a[a>0.5]
    # print(b)
    # a[a>0.5] = 0.5
    # print(a)

    # a = np.arange(0,60,10)
    # print('a =',a)
    # b = a.reshape((-1,1))
    # print(b)
    # c = np.arange(6)
    # print('c =',c)
    # f = b + c
    # print('f =\n',f)

    # a = np.arange(0, 60, 10).reshape((-1,1)) + np.arange(6)
    # print(a)
    # print(a[[0, 1, 2],[2, 3, 4]]) # 第0，1，2行，第2，3，4列的数
    # print(a[4,[2, 3, 4]])   # 第4行，第2，3，4列的数
    # print(a[4:,[2, 3, 4]])  # 第4行以后，第2，3，4列的数

    ## numpy与python数据库的时间比较
    # for j in np.logspace(0, 7, 8):
    #     N = 1000
    #     x = np.linspace(0, 10, j)
    #     start = time.clock()
    #     y = np.sin(x)
    #     t1 = time.clock() - start
    #
    #     x = x.tolist()
    #     start = time.clock()
    #     for i,t in enumerate(x):
    #         x[i] = math.sin(t)
    #     t2 = time.clock() - start
    #     print(j, ':', t1, t2, t2/t1)

    # a = np.array((1,2,3,4,5,6,7,7,6,3,4,6))
    # print(a)
    # b = np.unique(a)
    # print(b)

    # c = np.array(((1, 2), (3, 4), (5, 6), (1, 3), (3, 4), (7, 6)))
    # print(c)
    # # print(np.unique(c))
    # x = c[:,0] + c[:,1] * 1j
    # print('转为虚数后:',x)
    # print('虚数去重复后:',np.unique(x))
    # print(np.unique(x, return_index=True))
    # idx = np.unique(x, return_index=True)[1]
    # print('去重后的数组:',c[idx])

    # print('去重方法2：\n', np.array(list(set([tuple(t) for t in c]))))


    ## 4.3 stack and axis
    # a = np.arange(1, 10).reshape((3, 3))
    # b = np.arange(11, 20).reshape((3, 3))
    # c = np.arange(101, 110).reshape((3, 3))
    # print('a = \n', a)
    # print('b = \n', b)
    # print('c = \n', c)
    # print('axis = 0 \n', np.stack((a, b, c),axis=0))  # axis=0表示按集体堆叠
    # print('axis = 1 \n', np.stack((a, b, c), axis=1)) # axis=1表示按行堆叠
    # print('axis = 2 \n', np.stack((a, b, c), axis=2)) # axis=2表示按元素堆叠

    # a = np.arange(1, 10).reshape(3, 3)
    # print(a)
    # b = a + 10
    # print(b)
    # print(np.dot(a,b))
    # print(a * b)

    # a = np.arange(1, 10)
    # print(a)
    # b = np.arange(20, 25)
    # print(b)
    # print(np.concatenate((a, b)))  # 数组连接

    ##****画图*****
    # mpl.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
    # mpl.rcParams['axes.unicode_minus'] = False
    # mu = 0
    # sigma =1
    # x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 51)
    # y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    # print(x.shape)
    # print('x = \n', x)
    # print(y.shape)
    # print('y = \n', y)
    # plt.figure(facecolor='w') # 给定一个figure且背景取白色
    # plt.plot(x, y, 'r-', x, y, 'go', linewidth=2, markersize=8) # 使用坐标值画出红色实线且线宽为2l;使用坐标值画出绿色圆圈且圈的大小为8
    # plt.xlabel('X', fontsize=15)
    # plt.ylabel('Y', fontsize=15)
    # plt.title(u'高斯分布', fontsize=18)
    # plt.grid(True, linestyle='-.') # 画虚线格子
    # plt.show()

    # x = np.linspace(-2,3,1001,dtype=np.float)
    # y_logit = np.log(1 + np.exp((-x)))/ math.log(2)
    # x_boost = np.exp(-x)
    # y_01 = x < 0
    # y_hinge = 1.0 - x
    # y_hinge[y_hinge < 0] = 0
    # fig = plt.figure(figsize=(5,4))
    # plt.plot(x, y_logit,'r-',label='Logistic Loss',linewidth=2)
    # plt.plot(x,y_01,'g-',label='0-1 Loss',linewidth=2)
    # plt.plot(x,y_hinge,'b-',label='Hinge Loss',linewidth=2)
    # plt.plot(x, x_boost, 'm--', label='Hinge Loss', linewidth=2)
    # plt.grid(linestyle='-.')
    # plt.legend(loc='upper right')
    # # plt.savefig('1.png')
    # plt.show()

    # def f(x):
    #     y = np.ones_like(x) # 形成一个和x的形状相同，但内容填充为1的向量
    #     i = x > 0
    #     y[i] = np.power(x[i], x[i])
    #     i = x < 0
    #     y[i] = np.power(-x[i], -x[i])
    #     return y
    #
    #
    # x = np.linspace(-1.3, 1.3, 1001)
    # y = f(x)
    # plt.plot(x,y,'g-',label='x^x',linewidth=2)
    # plt.grid(linestyle='-.')
    # plt.legend(loc='upper left')
    # plt.show()

    # mpl.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
    # mpl.rcParams['axes.unicode_minus'] = False
    # x = np.arange(1,0,-0.01)
    # y = (-3 * x * np.log(x) + np.exp(-(40 * (x -1/np.e)) ** 4) / 25) / 2
    # plt.figure(figsize=(5,7))
    # plt.plot(y,x,'r-',linewidth=2)
    # plt.grid(True)
    # plt.title(u'胸型线',fontsize=20)
    # plt.show()

    # t = np.linspace(0,2*np.pi,100)
    # x = 16 * np.sin(t) ** 3
    # y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    # plt.plot(x,y,linewidth=2)
    # plt.grid(True)
    # plt.show()

    # t = np.linspace(0,50,num=1000)
    # x = t*np.sin(t) + np.cos(t)
    # y = np.sin(t) - t*np.cos(t)
    # plt.plot(x,y,'r-',linewidth=2)
    # plt.grid(True)
    # plt.show()

    # x = np.arange(0,10,0.1)
    # y = np.sin(x)
    # plt.bar(x,y,width=0.04,linewidth=0.2)
    # plt.plot(x,y,'r--',linewidth=2)
    # plt.xticks(rotation=-60)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.grid(linestyle='-.')
    # plt.show()

    # x = np.random.rand(10000)
    # t = np.arange(len(x))
    # # plt.hist(x,30,color='m',alpha=0.8,edgecolor='black')  # alpha为透明度
    # plt.plot(t,x,'g.')
    # plt.grid(True,linestyle='-.')
    # plt.show()

    # Poisson发布
    # x = np.random.poisson(lam=5,size=14)
    # pillar = 15
    # a = plt.hist(x,bins=range(1,pillar),normed=True,rwidth=0.8,color='g',alpha=0.5)
    # plt.grid(linestyle='-.')
    # plt.show()
    # print(a)

    # 3D图像
    # x,y = np.mgrid[-3:3:7j,-3:3:7j]
    # print(x)
    # print(y)
    u = np.linspace(-3,3,101)
    x,y = np.meshgrid(u,u)
    z = np.exp(-(x**2+y**2)/2)/math.sqrt(2*math.pi)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(x,y,z,cstride=3,rstride=3,edgecolor='black',cmap=cm.Accent,linewidth=0.5)
    plt.show()
    # # cmaps = [('Perceptually Uniform Sequential',
    # #           ['viridis', 'inferno', 'plasma', 'magma']),
    # #          ('Sequential', ['Blues', 'BuGn', 'BuPu',
    # #                          'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
    # #                          'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
    # #                          'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
    # #          ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
    # #                              'copper', 'gist_heat', 'gray', 'hot',
    # #                              'pink', 'spring', 'summer', 'winter']),
    # #          ('Diverging', ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
    # #                         'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
    # #                         'seismic']),
    # #          ('Qualitative', ['Accent', 'Dark2', 'Paired', 'Pastel1',
    # #                           'Pastel2', 'Set1', 'Set2', 'Set3']),
    # #          ('Miscellaneous', ['gist_earth', 'terrain', 'ocean', 'gist_stern',
    # #                             'brg', 'CMRmap', 'cubehelix',
    # #                             'gnuplot', 'gnuplot2', 'gist_ncar',
    # #                             'nipy_spectral', 'jet', 'rainbow',
    # #                             'gist_rainbow', 'hsv', 'flag', 'prism'])]
