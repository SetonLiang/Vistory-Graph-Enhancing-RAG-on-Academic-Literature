# 导入必要的库
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from wordcloud import WordCloud
import matplotlib.font_manager
print(matplotlib.matplotlib_fname())

# 构建词料库
corpus = [
    ['北京', '上海', '广州', '深圳', '成都', '西安', '杭州', '苏州', '厦门', '重庆'],
    ['故宫', '长城', '天安门', '颐和园', '西湖', '黄鹤楼', '张家界', '九寨沟', '三亚', '广州']
]

# 使用word2vec生成词向量
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

# 提取词向量
word_vectors = model.wv
word_vectors_list = [word_vectors[word] for word in model.wv.index_to_key]

# 使用PCA将词向量降到2维空间
pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(word_vectors_list)

# 尝试不同的聚类数量
best_num_clusters = 0
best_silhouette_score = -1
for num_clusters in range(2, 6):  # 尝试聚类数量从2到5
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(word_vectors_2d)  # 使用降维后的词向量进行聚类
    silhouette_avg = silhouette_score(word_vectors_2d, kmeans.labels_)
    print(f"For n_clusters = {num_clusters}, the average silhouette score is {silhouette_avg}")
    if silhouette_avg > best_silhouette_score:
        best_silhouette_score = silhouette_avg
        best_num_clusters = num_clusters

print(f"Best number of clusters: {best_num_clusters}")

# 使用最优簇数量进行聚类
kmeans = KMeans(n_clusters=best_num_clusters)
kmeans.fit(word_vectors_2d)  # 使用降维后的词向量进行聚类

# 输出聚类结果
for i in range(best_num_clusters):
    cluster_words = []
    for j in range(len(kmeans.labels_)):
        if kmeans.labels_[j] == i:
            cluster_words.append(word_vectors.index_to_key[j])
    print(f"Cluster {i+1}: {cluster_words}")

# 绘制散点图
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
for i in range(best_num_clusters):
    cluster_words = []
    cluster_vectors = []
    for j in range(len(kmeans.labels_)):
        if kmeans.labels_[j] == i:
            cluster_words.append(word_vectors.index_to_key[j])
            cluster_vectors.append(word_vectors_2d[j])
    cluster_vectors = np.array(cluster_vectors)
    plt.scatter(cluster_vectors[:, 0], cluster_vectors[:, 1], color=colors[i], label=f"Cluster {i+1}")
    for k, word in enumerate(cluster_words):
        plt.annotate(word, (cluster_vectors[k, 0], cluster_vectors[k, 1]))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Word Clusters Visualization')
plt.legend()
plt.show()

# 词云图可视化

all_cluster_words = []
for i in range(best_num_clusters):
    cluster_words = []
    for j in range(len(kmeans.labels_)):
        if kmeans.labels_[j] == i:
            cluster_words.append(word_vectors.index_to_key[j])
    all_cluster_words.extend(cluster_words)

all_cluster_text = " ".join(all_cluster_words)
print(all_cluster_text)
# 生成词云图
wordcloud = WordCloud(font_path='E://pycharm/Anaconda/anaconda/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/simhei.ttf',background_color='white')
wordcloud.generate(all_cluster_text)
wordcloud.to_file("1.png") # 保存词云文件
