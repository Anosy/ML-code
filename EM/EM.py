# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin


mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


if __name__ == '__main__':
    style = 'sklearn'

    # 构建两个类，三个特征的数据
    np.random.seed(0)
    mu1_fact = (0, 0, 0)
    cov1_fact = np.diag((1, 2, 3))
    data1 = np.random.multivariate_normal(mu1_fact, cov1_fact, 400)
    mu2_fact = (2, 2, 1)
    cov2_fact = np.array(((4, 1, 2), (2, 2, 1), (3, 1, 2)))
    data2 = np.random.multivariate_normal(mu2_fact, cov2_fact, 100)
    data = np.vstack((data1, data2))
    y = np.array([True] * 400 + [False] * 100)

    if style == 'sklearn':
        gmm = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=1000)
        gmm.fit(data)
        print('类别概率:\t', gmm.weights_[0])  # weights_ 表示每个类别的权值,  这里取了第一个类别的概率
        print('均值:\n', gmm.means_, '\n')  # 返回每个类别的每个特征的均值
        print('方差:\n', gmm.covariances_, '\n')  # 返回协方差   shape (n_components, n_features, n_features) if 'full'
        mu1, mu2 = gmm.means_
        sigma1, sigma2 = gmm.covariances_
    else:
        num_iter = 100
        n, d = data.shape  # data 为500维，三个特征的数据
        # 随机指定
        # mu1 = np.random.standard_normal(d)
        # print mu1
        # mu2 = np.random.standard_normal(d)
        # print mu2
        mu1 = data.min(axis=0)  # 初始化参数theta
        mu2 = data.max(axis=0)
        sigma1 = np.identity(d)
        sigma2 = np.identity(d)
        pi = 0.5
        # EM
        for i in range(num_iter):
            # E Step
            norm1 = multivariate_normal(mu1, sigma1)
            norm2 = multivariate_normal(mu2, sigma2)
            tau1 = pi * norm1.pdf(data)
            tau2 = (1 - pi) * norm2.pdf(data)
            gamma = tau1 / (tau1 + tau2)    # 求出对于每一个样本xi它由第k个组分生成的概率  这里计算的是由第一个组生成的概率

            # M Step
            mu1 = np.dot(gamma, data) / np.sum(gamma)
            mu2 = np.dot((1 - gamma), data) / np.sum((1 - gamma))
            sigma1 = np.dot(gamma * (data - mu1).T, data - mu1) / np.sum(gamma)
            sigma2 = np.dot((1 - gamma) * (data - mu2).T, data - mu2) / np.sum(1 - gamma)
            pi = np.sum(gamma) / n
            print(i, ":\t", mu1, mu2)
        print('类别概率:\t', pi)
        print('均值:\t', mu1, mu2)
        print('方差:\n', sigma1, '\n\n', sigma2, '\n')

    # 预测分类
    norm1 = multivariate_normal(mu1, sigma1)
    norm2 = multivariate_normal(mu2, sigma2)
    tau1 = norm1.pdf(data)
    tau2 = norm2.pdf(data)

    fig = plt.figure(figsize=(10, 5), facecolor='w')
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', s=30, marker='o', edgecolors='k', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('原始数据', fontsize=15)
    ax = fig.add_subplot(122, projection='3d')
    order = pairwise_distances_argmin([mu1_fact, mu2_fact], [mu1, mu2], metric='euclidean')
    print('order:', order)
    if order[0] == 0:
        c1 = tau1 > tau2
    else:
        c1 = tau1 < tau2
    c2 = ~c1
    acc = np.mean(y == c1)
    print('准确率：%.2f%%' % (100*acc))
    ax.scatter(data[c1, 0], data[c1, 1], data[c1, 2], c='r', s=30, marker='o', edgecolors='k', depthshade=True)
    ax.scatter(data[c2, 0], data[c2, 1], data[c2, 2], c='gmm', s=30, marker='^', edgecolors='k', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('EM算法分类', fontsize=15)
    plt.suptitle('EM算法的实现', fontsize=18)
    plt.subplots_adjust(top=0.90)
    plt.tight_layout()
    plt.show()
