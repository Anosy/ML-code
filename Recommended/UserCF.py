import sys
import random
import math
import os
from operator import itemgetter

from collections import defaultdict

random.seed(0)

class UserBasedCF(object):
    ''' TopN recommendation - User Based Collaborative Filtering '''
    ''' 构造一个UserCF协同过滤算法需要的流程
     1. 先将原始数据集进行训练集和测试集分割
     2. 计算用户之间的兴趣相似度
     3. 给user推荐
     4. 评测
    '''
    def __init__(self):
        self.trainset = {}   # 训练集
        self.testset = {}    # 测试集

        self.n_sim_user = 20   # 用户相同的个数
        self.n_rec_movie = 10  # 推荐的电影的个数

        self.user_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

        print ('Similar user number = %d' % self.n_sim_user, file=sys.stderr)
        print ('recommended movie number = %d' %
               self.n_rec_movie, file=sys.stderr)   # sys.stderrr 为了让输出变成红字。或者说是为了在cmd上能看到输出日志

    @staticmethod
    def loadfile(filename):  # 载入文件生成迭代器
        ''' load a file, return a generator. '''
        fp = open(filename, 'r')  # 打开文件
        for i, line in enumerate(fp):
            yield line.strip('\r\n')   # 将每行生成迭代器返回
            if i % 100000 == 0:
                print ('loading %s(%s)' % (filename, i), file=sys.stderr)  # 每加载100000打印结果
        fp.close()
        print ('load %s succ' % filename, file=sys.stderr)        # 数据加载完成

    def generate_dataset(self, filename, pivot=0.7):  # 产生数据，默认将按7/3来进行数据集的分割
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0

        for line in self.loadfile(filename):  # 取文件的每一行
            user, movie, rating, _ = line.split('::') # 将文件按照“::”来切分得到user, movie, rating
            # split the data by pivot
            if random.random() < pivot:
                self.trainset.setdefault(user, {})    # 得到trainset={user:{}}如果终点中包含给定的键，则返回该键对应的值，否则返回该键设置的值。以为trainset为空的字典，所以这里就是创建嵌套字典
                self.trainset[user][movie] = int(rating)  # 得到 trainset={user:{movie:rating}}
                trainset_len += 1                       # 记录训练集的长度
            else:
                self.testset.setdefault(user, {})     # 同训练集
                self.testset[user][movie] = int(rating)
                testset_len += 1

        print ('split training set and test set succ', file=sys.stderr)
        print ('train set = %s' % trainset_len, file=sys.stderr)
        print ('test set = %s' % testset_len, file=sys.stderr)

    def calc_user_sim(self):  # 计算用户之间的兴趣相似度
        ''' calculate user similarity matrix '''  # 计算用户兴趣相似度矩阵
        # build inverse table for item-users         # 构建倒排表
        # key=movieID, value=list of userIDs who have seen this movie
        print ('building movie-users inverse table...', file=sys.stderr)
        movie2users = dict()

        for user, movies in self.trainset.items():  # 获取用户和电影
            for movie in movies:
                # inverse table for item-users
                if movie not in movie2users:
                    movie2users[movie] = set()
                movie2users[movie].add(user)    # 得到电影-用户表
                # count item popularity at the same time
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1  # 计算每个电影在训练集有多少人操作过
        print ('build movie-users inverse table succ', file=sys.stderr)

        # save the total movie number, which will be used in evaluation
        self.movie_count = len(movie2users)    # 电影的数量
        print ('total movie number = %d' % self.movie_count, file=sys.stderr)

        # count co-rated items between users
        usersim_mat = self.user_sim_mat
        print ('building user co-rated movies matrix...', file=sys.stderr)

        for movie, users in movie2users.items():
            for u in users:
                usersim_mat.setdefault(u, defaultdict(int))  # 构建字典{user1:{user2:喜欢同一个电影的次数}}
                for v in users:
                    if u == v:
                        continue
                    usersim_mat[u][v] += 1 /math.log(1+len(users))
        print ('build user co-rated movies matrix succ', file=sys.stderr)

        # calculate similarity matrix
        print ('calculating user similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000

        for u, related_users in usersim_mat.items():
            for v, count in related_users.items():
                usersim_mat[u][v] = count / math.sqrt(len(self.trainset[u]) * len(self.trainset[v]))  # 用户u，v的相似度=用户共同喜欢的电影数/(用户u的电影数+用户v的电影数)
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:    # 每计算2000000，打印输出
                    print ('calculating user similarity factor(%d)' %
                           simfactor_count, file=sys.stderr)

        print ('calculate user similarity matrix(similarity factor) succ',
               file=sys.stderr)
        print ('Total similarity factor number = %d' %
               simfactor_count, file=sys.stderr)

    def recommend(self, user):
        ''' Find K similar users and recommend N movies. '''
        K = self.n_sim_user   # 取前K个相同用户
        N = self.n_rec_movie  # 推荐的N个电影
        rank = dict()
        watched_movies = self.trainset[user]  # 用户实际观看过的电影

        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(), key=itemgetter(1), reverse=True)[0:K]:  # 取共同次数最多的K个用户和共同的次数
            for movie in self.trainset[similar_user]:  # 取K个用户看过的电影
                if movie in watched_movies:  #  如果该电影用户已经看过就不进行操作
                    continue
                rank.setdefault(movie, 0)  # 初始化rank字典，所有的新键默认的值都设置为0
                rank[movie] += similarity_factor# rank={movie:兴趣度}
        # return the N best movies
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]  # 将所有的推荐的电影，按照兴趣度的大小排列，取出前N个

    def evaluate(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print ('Evaluation start...', file=sys.stderr)

        N = self.n_rec_movie  # 推荐的电影的个数
        #  varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_movies = set()
        # varables for popularity
        popular_sum = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print ('recommended for %d users' % i, file=sys.stderr)
            test_movies = self.testset.get(user, {})  # 获取当前用户测试集的电影列表
            rec_movies = self.recommend(user)  # 得到用户的推荐电影列表
            for movie, _ in rec_movies:
                if movie in test_movies:
                    hit += 1   # 如果推荐的电影列表中又电影和测试集的列表中的电影相同，那么就算命中一次，hit+1
                all_rec_movies.add(movie)  # 记录所有用户的全部的推荐电影
                popular_sum += math.log(1 + self.movie_popular[movie])  # 计算当前用户推荐的所有电影的流行度
            rec_count += N  # 记录一共推荐了多少电影
            test_count += len(test_movies)  # 获取所有用户对电影发生过评价的数量

        precision = hit / (1.0 * rec_count)  # 测试集所有命中的次数/总共推荐了多少电影
        recall = hit / (1.0 * test_count)  # 测试集所有命中的次数/所有用户的电影发生过评价的个数
        coverage = len(all_rec_movies) / (1.0 * self.movie_count) # 推荐的所有电影/训练集的所有电影总数
        popularity = popular_sum / (1.0 * rec_count)  # 计算平均流行度

        print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
               (precision, recall, coverage, popularity), file=sys.stderr)


if __name__ == '__main__':
    ratingfile = os.path.join('ml-1m', 'ratings.dat')
    usercf = UserBasedCF()
    usercf.generate_dataset(ratingfile)  # 产生训练数据和测试数据
    usercf.calc_user_sim()
    usercf.evaluate()
