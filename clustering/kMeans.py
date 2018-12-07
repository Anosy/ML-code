import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # 1.0.x 以上版本绘制3D图 需要导入


def expand(a, b):
    d = (b - a) * 0.1
    return a-d, b+d

if __name__ == '__main__':
    N = 400
    centers = 4
    data, y = ds.make_blobs(N, n_features=3, centers=centers, random_state=3)
    data1, y1 = ds.make_blobs(N, n_features=3, centers=centers, cluster_std=[1,2.5,0.6,2],random_state=3)
    data2 = np.vstack([data[y==0][:], data[y==1][:50],data[y == 2][:20], data[y == 3][:5]])
    y2 = np.array([0] * 100 + [1] * 50 + [2] * 20 + [3] * 5)
    m = np.array(((1, 1, 1), (1, 3, 2), (2,3,1)))
    data_r = data.dot(m)

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    cm = matplotlib.colors.ListedColormap(list('rgbm'))

    data_list = data, data, data_r, data_r, data1, data1, data2, data2
    y_list = y, y, y, y, y1, y1, y2, y2
    titles = '原始数据', 'KMeans++聚类', '旋转后数据', '旋转后KMeans++聚类', \
             '方差不相等数据', '方差不相等KMeans++聚类', '数量不相等数据', '数量不相等KMeans++聚类'

    model = KMeans(n_clusters=4, init='k-means++', n_init=5)  # 簇为4，使用k-means++， 将结果做5次，选择结果最佳的作为最终模型
    fig = plt.figure(figsize=(8, 9), facecolor='w')
    for i,(x,y,title) in enumerate(zip(data_list, y_list, titles), start=1):
        # plt.subplot(4,2,i)
        ax = fig.add_subplot(4,2,i, projection='3d')
        plt.title(title)
        if i % 2 == 1:
            y_pred = y
        else:
            y_pred = model.fit_predict(x)
        ax.scatter(x[:,0], x[:,1], x[:,2],s=10, c=y_pred, cmap=cm, edgecolors='none')
        # x1_min, x2_min = np.min(x, axis=0)
        # x1_max, x2_max = np.max(x, axis=0)
        # x1_min, x1_max = expand(x1_min, x1_max)
        # x2_min, x2_max = expand(x2_min, x2_max)  # 让图像展示的更加完整
        # plt.xlim((x1_min, x1_max))
        # plt.ylim((x2_min, x2_max))
        plt.grid(b=True, ls=':')
    plt.tight_layout(2, rect=(0, 0, 1, 0.97))
    plt.suptitle('数据分布对KMeans聚类的影响', fontsize=18)
    plt.show()

