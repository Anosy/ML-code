from gensim import corpora, models, similarities
from pprint import pprint
import numpy as np
import warnings


if __name__ == '__main__':
    f = open('LDA_test.txt')
    stop_list = set('for a of the and to in'.split())

    # 获取每个文本的词语，且除去停用词
    print('After')
    texts = [[word for word in line.strip().lower().split() if word not in stop_list]for line in f]
    print('Text = ')
    pprint(texts)

    # 将每个文章中的词，根据字典转化为bag-of-word形式
    dictionary = corpora.Dictionary(texts)  # 提取文档的非重复词语作为字典
    print(dictionary)
    V = len(dictionary)
    corpus = [dictionary.doc2bow(text) for text in texts]  # 对每篇文档，根据字典将其转化为bag-of-word形式
    pprint(corpus)

    # 将bag-of-word形式，输入到TF-IDF模型中，将其转化为TF-IDF格式
    model = models.TfidfModel(corpus)
    corpus_tfidf = model[corpus]
    for each in corpus_tfidf:
        print(each)

    # Lsi模型
    print('\nLSI Model:')
    lsi = models.LsiModel(corpus_tfidf, num_topics=2, id2word=dictionary)
    print(lsi.get_topics())
    topic_result = [a for a in lsi[corpus_tfidf]]
    pprint(topic_result)  # 输出每篇文档对应的主题的概率

    print('LSI Topics:')
    # pprint(lsi.print_topics(num_topics=2, num_words=5))
    for i in range(2):
        print(lsi.show_topic(i))  # 输出文章的每一个主题中最重要的词，以及其对主题的贡献，其实质是左奇异矩阵的列向量

    similarity = similarities.MatrixSimilarity(lsi[corpus_tfidf])  # similarities.Similarity()
    print('Similarity:')
    pprint(list(similarity))  # 计算每篇文档的相似度

    # LDA模型
    print('\nLDA Model:')
    num_topics = 2
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                          alpha='auto', eta='auto', minimum_probability=0.001, passes=10)
    doc_topic = [doc_t for doc_t in lda[corpus_tfidf]]
    print('Document-Topic:\n')
    pprint(doc_topic)  # 输出每篇文档对应的主题的概率

    for doc_topic in lda.get_document_topics(corpus_tfidf): # 输出每篇文档对应的主题的概率，同上
        print(doc_topic)

    for topic_id in range(num_topics):  # 输出文章的每一个主题中最重要的词，以及其对主题的贡献
        print('Topic', topic_id)
    #     # pprint(lda.get_topic_terms(topicid=topic_id))
        pprint(lda.show_topic(topic_id))

    # 输出与第零篇文档最相近的文档
    similarity = similarities.MatrixSimilarity(lda[corpus_tfidf])  # 计算9篇文档的相似度
    sim_zero = np.array(similarity)[0].argsort()[1]
    print('Most similarity_doc: %d ' % sim_zero)
    print(texts[sim_zero])