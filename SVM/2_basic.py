import numpy as np
from scipy import stats
import math
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.special import gamma
from scipy.special import factorial
import os
from PIL import Image
from pprint import pprint


def calc_statistics(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    skew = stats.skew(x)
    kurtosis = stats.kurtosis(x)
    return mu, sigma, skew, kurtosis


def restore1(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for k in range(K):
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        a += sigma[k] * np.dot(uk, vk)
    a[a < 0] = 0
    a[a > 255] = 255
    # a = a.clip(0, 255)
    return np.rint(a).astype('uint8')

if __name__ == '__main__':

    # d = np.random.randn(100000)  # 高斯发布，且取100000个样本点
    # mu, sigma, skew, kurtosis = calc_statistics(d)
    # print('函数库计算均值、标准差、偏度、峰度：', mu, sigma, skew, kurtosis)
    # mpl.rcParams[u'font.sans-serif'] = 'SimHei'
    # mpl.rcParams[u'axes.unicode_minus'] = False
    # y1, x1, dummy = plt.hist(d, bins=50, normed=1, color='g', alpha=0.75,edgecolor='black')  # bin为柱状图的条数,norm指定密度,也就是每个条状图的占比例比
    # t = np.arange(x1.min(), x1.max(), 0.05)
    # # y = np.exp(-t ** 2 / 2) / math.sqrt(2 * math.pi)
    # y = mlab.normpdf(x1, mu, sigma)
    # plt.plot(x1, y, 'r-', lw=2)
    # plt.title(u'高斯分布，样本个数：%d' % d.shape[0])
    # plt.grid(True)
    # plt.axis([-4, 4, 0, 0.4]) # 设置坐标轴取值范围
    # plt.show()

    # 二维图像
    # d = np.random.randn(100000, 2)
    # mu, sigma, skew, kurtosis = calc_statistics(d)
    # N = 30
    # density, edges = np.histogramdd(d, bins=[N, N])
    # density /= density.max()
    # x = y = np.arange(N)
    # t = np.meshgrid(x, y)
    # mpl.rcParams[u'font.sans-serif'] = 'SimHei'
    # mpl.rcParams[u'axes.unicode_minus'] = False
    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')
    # ax.scatter(t[0], t[1], density, c='r', s=15 * density, marker='o', depthshade=True,edgecolor='black')
    # ax.plot_surface(t[0], t[1], density, cmap=cm.Accent, rstride=2, cstride=2, alpha=0.9, lw=0.75,edgecolor='black')
    # ax.set_xlabel(u'X')
    # ax.set_ylabel(u'Y')
    # ax.set_zlabel(u'Z')
    # plt.title(u'二元高斯分布，样本个数：%d' % d.shape[0], fontsize=20)
    # plt.tight_layout(0.1) # 调整图像边缘及图像间的空白间隔
    # plt.show

    # gamma
    # mpl.rcParams['axes.unicode_minus'] = False
    # mpl.rcParams['font.sans-serif'] = 'SimHei'
    # N = 5
    # x = np.linspace(0, N, 50)
    # y = gamma(x+1)
    # plt.figure()
    # plt.plot(x,y,'r-',x,y,'m^',lw=2)
    # z = np.arange(0,N+1)
    # f = factorial(z, exact=True)  # 计算阶乘
    # plt.plot(z, f, 'go', markersize=8)
    # plt.grid(b=True)
    # plt.xlim(-0.1, N + 0.1)
    # plt.ylim(0.5, np.max(y) * 1.05)
    # plt.xlabel(u'X', fontsize=15)
    # plt.ylabel(u'Gamma(X) - 阶乘', fontsize=15)
    # plt.title(u'阶乘和Gamma函数', fontsize=16)
    # plt.show()

    # SVD 奇异值分解
    A = Image.open('lena.png','r')
    output_path = r'.\Pic'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    a = np.array(A)
    K = 12
    u_r, sigma_r, v_r = np.linalg.svd(a[:, :, 0])
    u_g, sigma_g, v_g = np.linalg.svd(a[:, :, 1])
    u_b, sigma_b, v_b = np.linalg.svd(a[:, :, 2])
    plt.figure(figsize=(10, 10), facecolor='w')
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    for k in range(1, K + 1):
        print(k)
        R = restore1(sigma_r, u_r, v_r, k)
        G = restore1(sigma_g, u_g, v_g, k)
        B = restore1(sigma_b, u_b, v_b, k)
        I = np.stack((R, G, B), axis=2)
        # Image.fromarray(I).save('%s\\svd_%d.png' % (output_path, k))
        if k <= 12:
            plt.subplot(3, 4, k)
            plt.imshow(I)
            plt.axis('off')  # 取消坐标轴显示
            plt.title(u'奇异值个数：%d' % k)
    plt.suptitle(u'SVD与图像分解', fontsize=20)
    plt.tight_layout(0.3, rect=(0, 0, 1, 0.92))
    # plt.subplots_adjust(top=0.9)
    plt.show()





