import numpy as np
from gensim import corpora, models, similarities
from pprint import pprint
import time


def load_stopword():
    f_stop = open('./stopword.txt', encoding='GBK')
    sw = [line.strip() for line in f_stop]
    f_stop.close()
    return sw

def load_stopword1():
    with open('./stopword.txt') as f:
        sw = [line.strip() for line in f]
    return sw

if __name__ == '__main__':
    print('初始化停止词列表 --')
    t_start = time.time()
    stop_words = load_stopword()

    print('开始读入语料数据 -- ')
    f = open('./news.dat', encoding='utf-8')  # LDA_test.txt
    texts = [[word for word in line.strip().lower().split() if word not in stop_words] for line in f]
    # texts = [line.strip().split() for line in f]
    print('读入语料数据完成，用时%.3f秒' % (time.time() - t_start))
    f.close()
    M = len(texts)
    print('文本数目：%d个' % M)  # 文章的个数

    print('正在建立词典 --')
    dictionary = corpora.Dictionary(texts)
    V =  len(dictionary)
    print('词的个数：', V)   # 字典中词的个数

    print('正在计算文本向量 --')  # # 对每篇文档，根据字典将其转化为bag-of-word形式
    corpus = [dictionary.doc2bow(text) for text in texts]

    print('正在计算文档TF-IDF --')
    t_start = time.time()
    model = models.TfidfModel(corpus)
    corpus_tfidf = model[corpus]
    print('建立文档TF-IDF完成，用时%.3f秒' % (time.time() - t_start))

    print('LDA模型拟合推断 --')
    num_topics = 10
    t_start = time.time()
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                          alpha=0.01, eta=0.01, minimum_probability=0.001,
                          update_every=1, chunksize=100, passes=5)
    print('LDA模型完成，训练时间为\t%.3f秒' % (time.time() - t_start))

    # 随机打印某10个文档的主题
    num_show_topic = 10  # 每个文档显示前几个主题
    print('10个文档的主题分布：')
    doc_topics = lda.get_document_topics(corpus_tfidf)  # 所有文档的主题分布
    idx = np.arange(M)
    np.random.shuffle(idx)
    idx = idx[:10]
    for i in idx:
        topic = np.array(doc_topics[i])
        print('topic = \n', topic)
        topic_distribute = np.array(topic[:, 1])
        # print topic_distribute
        topic_idx = topic_distribute.argsort()[:-num_show_topic - 1:-1]
        print(('第%d个文档的前%d个主题：' % (i, num_show_topic)), topic_idx)
        print(topic_distribute[topic_idx])

    num_show_term = 7  # 每个主题显示几个词
    print('每个主题的词分布：')
    for topic_id in range(num_topics):
        print('主题#%d：\t' % topic_id)
        term_distribute_all = lda.get_topic_terms(topicid=topic_id)
        term_distribute = term_distribute_all[:num_show_term]
        term_distribute = np.array(term_distribute)
        term_id = term_distribute[:, 0].astype(np.int)
        print('词：\t', end=' ')
        for t in term_id:
            print(dictionary.id2token[t], end=' ')  # 将得到的id给转化为词
        print()

    # similarity = similarities.MatrixSimilarity(lda[corpus_tfidf])  # 计算9篇文档的相似度
    # print('Similarity:')
    # pprint(list(similarity))
    # 
    # # 输出与第零篇文档最相近的文档
    # similarity = similarities.MatrixSimilarity(lda[corpus_tfidf])  # 计算9篇文档的相似度
    # sim_zero = np.array(similarity)[0].argsort()[1]
    # print('Most similarity_doc: %d ' % sim_zero)
    # print(texts[0])
    # print(texts[sim_zero])