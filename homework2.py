import jieba
from gensim import corpora, models, similarities
from collections import defaultdict  # 用于创建一个空的字典，在后续统计词频中可清理词频少的词语

# 读取文档
doc1 = "data1.txt"
doc2 = "data2.txt"
doc3 = "data3.txt"
d1 = open(doc1, encoding='GBK').read()
d2 = open(doc2, encoding='GBK').read()
d3 = open(doc3, encoding='GBK').read()

# 对要计算的文档进行分词
data1 = jieba.cut(d1)
data2 = jieba.cut(d2)
data3 = jieba.cut(d3)

# 对分词完的数据进行整理为指定格式
data11 = " ".join(data1)
data21 = " ".join(data2)
data31 = " ".join(data3)
documents = [data11, data21, data31]
# print(documents)
texts = [[word for word in document.split()] for document in documents]

# 计算词语的频率
frequency = defaultdict(int)
for text in texts:
    for word in text:
        frequency[word] += 1

# 对频率低的词语进行过滤
texts = [[word for word in text if frequency.get(word, 0) > 2] for text in texts]

# 通过语料库将文档的词语进行建立词典
dictionary = corpora.Dictionary(texts)
dictionary.save("dict.txt")  # 可以将生成的词典进行保存


# 将要对比的文档通过doc2bow转化为稀疏向量
new_xs1 = dictionary.doc2bow(data11.split())
new_xs2 = dictionary.doc2bow(data21.split())
new_xs3 = dictionary.doc2bow(data31.split())

# 对语料库进一步处理，得到新语料库
corpus = [dictionary.doc2bow(text) for text in texts]
# print(corpus)

# 将新语料库通过tf-idf model 进行处理，得到tf-idf
tfidf = models.TfidfModel(corpus)

# 通过token2id得到特征数
featurenum = len(dictionary.token2id.keys())

# 稀疏矩阵相似度，从而建立索引
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features = featurenum)

# 得到最终相似结果
sim = index[tfidf[new_xs1]]
print("文档1与文档2相似度为：", sim[1])
print("文档1与文档3相似度为：", sim[2])
sim = index[tfidf[new_xs2]]
print("文档2与文档3相似度为：", sim[2])