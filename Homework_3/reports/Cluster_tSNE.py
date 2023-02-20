#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join
from IPython import display
from sklearn.datasets import load_digits
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score # и другие метрики
from sklearn.cluster import KMeans # а также другие алгоритмы


# In[2]:


DATA_PATH = "data"
plt.rcParams["figure.figsize"] = 12, 9
sns.set_style("whitegrid")
warnings.filterwarnings("ignore")

SEED = 111
random.seed(SEED)
np.random.seed(SEED)


# ### Задание 1. Реализация Kmeans
# 
# 5 баллов

# В данном задании вам предстоит дописать код класса `MyKMeans`. Мы на простом примере увидим, как подбираются центры кластеров и научимся их визуализировать.

# Сгенерируем простой набор данных, 400 объектов и 2 признака (чтобы все быстро работало и можно было легко нарисовать):

# In[3]:


X, true_labels = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])


# Напишем функцию `visualize_clusters`, которая по данным и меткам кластеров будет рисовать их и разукрашивать:

# In[4]:


def visualize_clusters(X, labels):
    """
    Функция для визуализации кластеров
        :param X: таблица объекты х признаки
        :param labels: np.array[n_samples] - номера кластеров
    """
    
#     print(X[:, 0], X[:, 1])
    
    unique_labels = np.sort(np.unique(labels))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, 
                    palette="colorblind", legend=False,
                    hue_order=unique_labels)
    plt.xlabel("$X_1$", fontsize=18)
    plt.ylabel("$X_2$", fontsize=18)
    
    for label in labels:
        center = X[(labels == label)].mean(axis=0)
        plt.scatter(x=center[0], y=center[1], s=80, c="#201F12", marker=(5, 2))


# In[5]:


visualize_clusters(X, true_labels)


# Напишем свой класс `MyKMeans`, который будет реализовывать алгоритм кластеризации K-средних. Напомним сам алгоритм:
# 
# 1. Выбераем число кластеров (K)
# 2. Случайно инициализируем K точек (или выбираем из данных), это будут начальные центры наших кластеров
# 3. Далее для каждого объекта считаем расстояние до всех кластеров и присваиваем ему метку ближайщего
# 4. Далее для каждого кластера считаем "центр масс" (среднее значение для каждого признака по всем объектам кластера)
# 5. Этот "центр масс" становится новым центром кластера
# 6. Повторяем п.3, 4, 5 заданное число итераций или до сходимости
# 
# Во время предсказания алгоритм просто находит ближайщий центроид (центр кластера) для тестового объекта и возвращает его номер.

# Реализуйте методы:
# * `_calculate_distance(X, centroid)` - вычисляет Евклидово расстояние от всех объектов в `Х` до заданного центра кластера (`centroid`)
# * `predict(X)` - для каждого элемента из `X` возвращает номер кластера, к которому относится данный элемент

# In[6]:


class MyKMeans:
    def __init__(self, n_clusters, init="random", max_iter=300, visualize=False):
        """
        Конструктор класса MyKMeans
            :param n_clusters: число кластеров
            :param init: способ инициализации центров кластеров
                'random' - генерирует координаты случайно из нормального распределения
                'sample' - выбирает центроиды случайно из объектов выборки
            :param max_iter: заданное число итераций 
                (мы не будем реализовывать другой критерий остановки)
            :param visualize: рисовать ли кластеры и их центроиды в процессе работы
                код будет работать сильно дольше, но красиво...
        """
        
        assert init in ["random", "sample"], f"Неизвестный метод инициализации {init}"
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None
        self.visualize = visualize
       
    
    def fit(self, X):
        """
        Подбирает оптимальные центры кластеров
            :param X: наши данные (n_samples, n_features)
        :return self: все как в sklearn
        """
        
        n_samples, n_features = X.shape
        
        # Инициализация центров кластеров
        if self.init == "random":
            centroids = np.random.randn(self.n_clusters, n_features)
        elif self.init == "sample":
            centroids_idx = np.random.choice(np.arange(n_samples), 
                                             size=self.n_clusters, 
                                             replace=False)
            centroids = X[centroids_idx]
        
        # Итеративно двигаем центры
        for _ in range(self.max_iter):
            # Посчитаем расстояния для всех объектов до каждого центроида
            dists = []
            for centroid in centroids:
                dists.append(self._calculate_distance(X, centroid))
            dists = np.concatenate(dists, axis=1)
            # Для каждого объекта найдем, к какому центроиду он ближе
            cluster_labels = np.argmin(dists, axis=1)
            
            # Пересчитаем центр масс для каждого кластера
            centroids = []
            for label in np.sort(np.unique(cluster_labels)):
                center = X[(cluster_labels == label)].mean(axis=0)
                centroids.append(center)
            
            # Отрисуем точки, покрасим по меткам кластера, а также изобразим центроиды
            if self.visualize:
                visualize_clusters(X, cluster_labels)
                display.clear_output(wait=True)
                display.display(plt.gcf())
                plt.close()
                
        self.centroids = np.array(centroids)
        
        return self
    
    
    def predict(self, X):
        """
        Для каждого X возвращает номер кластера, к которому он относится
            :param X: наши данные (n_samples, n_features)
        :return cluster_labels: метки кластеров
        """
        
        dists = []
        for centroid in self.centroids:
            dists.append(self._calculate_distance(X, centroid))
        
        dists = np.concatenate(dists, axis=1)
        
        cluster_labels = np.argmin(dists, axis=1)
        
        return cluster_labels
        
        
    def _calculate_distance(self, X, centroid):
        """
        Вычисляет Евклидово расстояние от всех объектов в Х до заданного центра кластера (centroid)
            :param X: наши данные (n_samples, n_features)
            :param centroid: координаты центра кластера
        :return dist: расстояния от всех X до центра кластера
        """
        
        dist = np.sum((X[:, None] - centroid)**2, axis=-1)
        
        return dist
    
    
    def __repr__(self):
        return f"Привет, я твой KMeans (/¯◡ ‿ ◡)/¯☆*"


# Обучите `MyKMeans` на наших игручешных данных, добейтесь сходимости. Не забудьте поставить `visualize=True`, чтобы посмотреть на красивые картинки. Также попробуйте различные способы инициализации центроидов и скажите, какой лучше подошел в этой ситуации.

# In[7]:


kmeans_random = MyKMeans(4, init="random")


# In[8]:


kmeans_random.fit(X)


# In[9]:


pred_labels_random = kmeans_random.predict(X)


# ### Not random

# In[10]:


kmeans = MyKMeans(4, init="sample")
kmeans.fit(X)


# In[11]:


pred_labels = kmeans.predict(X)


# ### Score

# In[12]:


from sklearn.metrics import accuracy_score, f1_score


# In[13]:


print(f"F1 score for random = {f1_score(true_labels, pred_labels_random, average='macro')}")


# In[14]:


print(f"Accuarcy for random = {accuracy_score(true_labels, pred_labels_random)}")


# In[15]:


print(f"F1 score = {f1_score(true_labels, pred_labels, average='macro')}")


# In[16]:


print(f"Accuarcy= {accuracy_score(true_labels, pred_labels)}")


# **Очевидно, что инициализация _sample_ лучше**

# ### посмотрим как это происходит на примере 5 итераций

# In[17]:


kmeans = MyKMeans(4, init="sample", visualize=True, max_iter=5)
kmeans.fit(X)


# ### Задание 2. Подбираем лучшую иерархическую кластеризацию
# 
# 5 баллов

# На лекции были рассмотрены различные расстояния, которые могут служить метриками различия между объектами. Также мы разобрали несколько алгоритмов кластеризации, в том числе и иерархическую. Часто может быть непонятно, какой алгоритм и какую метрику расстояния нужно взять. Давайте упростим себе жизнь и напишем функцию `algorithm_selection`, которая будет на основании переданных ей:
# 
# * метрик расстояния (можно брать все, что было на лекциях, минимум 4)
# * параметра `linkage` ('average', 'single', 'complete')
# * и метрик качества кластеризации ('Homogeneity', 'Completeness', 'V-measure', 'Silhouette')
# 
# будет выводить итоговую таблицу, в которую войдут столбцы:
# * distance (метрика расстояния)
# * linkage (алгоритм иерархической кластеризации)
# * homogenity
# * completeness
# * v_measure
# * silhouette
# 
# В результате по этой таблице, нужно сделать вывод о том, какой алгоритм кластеризации и с какими гиперпараметрами лучше всего сработал.

# Загрузим наши данные:

# In[18]:


data = load_digits()
X, y = data.data, data.target


# In[19]:


plt.imshow(X[0].reshape(8, 8).astype(int), cmap="gray")
plt.axis("off");


# Работать будем с изображениями рукописных цифр. Тут все похоже на данные для 1 домашнего задания, каждая картинка представлена вектором из 64 элементов (числа от 0 до 255). Чтобы ее нарисовать мы должны сделать `reshape` в картинку 8 на 8 пикселей. Вам нужно будет выбрать наилучший способ кластеризации при помощи функции `algorithm_selection`, которую вы реализуете. Для некоторых метрик кластеризации требуются метки **классов** объектов (они хранятся в переменной `y`).

# In[20]:


pd.Series(y).nunique()


# **Значит будет 10 кластеров**

# ##### YOUR TURN TO CODE

# In[21]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, silhouette_score


# In[22]:


distances = ["euclidean", "l2", "manhattan", "cosine"]
linkages = [ 'complete', 'average', 'single']
metrics = [homogeneity_score, completeness_score, v_measure_score, silhouette_score]


# In[23]:


def algorithm_selection(X, y, distances, algorithms, metrics):
    """
    Для заданных алгоримов кластеризации и гиперпараметров 
    считает различные метрики кластеризации
        :param X: наши данные (n_samples, n_features)
        :param distances: список возможных метрик расстояния
        :param algorithm: параметр linkage ('average', 'single', 'complete')
        :param metrics: список возможных метрик качества кластеризации
    :return compare_dataframe: таблица с метриками кластеризации
    """
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    df_list = []
    
    for linkage in algorithms:
        
        for distance in distances:
            
            clusterer = AgglomerativeClustering(n_clusters=10, affinity=distance, linkage=linkage)
            y_pred = clusterer.fit_predict(X_test)
            
            current_row =  [linkage, distance]
            
            for metric in metrics:
                
                if metric == silhouette_score:
                    current_row.append(metric(X_test, y_pred))
                else:
                    current_row.append(metric(y_test, y_pred))
        
            df_list.append(current_row)
                
    
    compare_dataframe = pd.DataFrame(df_list ,columns=['Linkage', "Distance",
                           "homogeneity_score", "completeness_score",
                           "v_measure_score", "silhouette_score"]).\
    sort_values(by = ["homogeneity_score", "completeness_score",
                           "v_measure_score", "silhouette_score"], ascending=False)
                
    return compare_dataframe


# In[24]:


compare_df = algorithm_selection(X, y, distances=distances, algorithms=linkages, metrics=metrics)
compare_df


# #### Кажется, что алгоритм average оказывается лучше всех, особенно с манхэттеновским и L2 нормой)

# ### Задание 3. Аннотация клеточных типов
# 
# суммарно 10 баллов

# В этом задании вам предстоит применять полученные знания о кластеризации для аннотации данных, полученных при помощи проточной цитометрии. Каждая строка в данных это одна клетка, столбцы **FSC-A**, **SSC-A** косвенным образом свидетельствуют о размере клеток, остальные показывают интенсивность флуоресценции различных поверхностных маркеров. Ваша задача определить для каждой клетки, к какому типу она относится.

# #### 3.1. EDA
# 
# 1.5 балла
# 
# Проведите EDA:
# 
# 1. Посмотрите на данные (можно попробовать метод `describe`)
# 2. Сколько у нас клеток / признаков
# 3. Постройте распределения каждого признака (есть ли очевинные выбросы?)

# In[25]:


fc_data = pd.read_csv(join(DATA_PATH, "flow_c_data.csv"), index_col=0).rename(columns = {'FSC-A':'FSC_A', 'SSC-A':'SSC_A',
                                                                                        "HLA-DR":"HLA_DR"})
fc_data.head()


# In[26]:


fc_data.describe()


# In[27]:


fc_data.info()


# In[28]:


print(f'Всего {fc_data.shape[0]} клеток и {fc_data.shape[1]} признаков')


# In[29]:


fig, axs = plt.subplots(3, 3, figsize=(10, 10))

for i in range(3):
    for j in range(3):
        sns.histplot(x=fc_data.iloc[:, i + j * 3], ax=axs[i, j], binwidth=1, kde=True)


# **Очевидно, что выбросы есть, особенно в FSC-A и SSC-A**

# In[30]:


fc_data.loc[fc_data.FSC_A > 5]


# In[31]:


fc_data = fc_data.loc[fc_data.FSC_A < 5]


# In[32]:


fig, axs = plt.subplots(3, 3, figsize=(10, 10))

for i in range(3):
    for j in range(3):
        sns.histplot(x=fc_data.iloc[:, i + j * 3], ax=axs[i, j], binwidth=1, kde=True)


# **Больше убирать выбросы не рискну, так как можно потерять нужную информацию**

# #### 3.2. Кластеризация
# 
# 4.5 балла

# При ручной аннотации клеточных типов обычно поступают следующим образом:
# 
# 1. При помощи методов понижения размерности рисуют наши наблюдения, чтобы примерно оценить число клеточных типов
# 2. Проводят кластеризацию наблюдений (для некоторых методов нужно заранее задать число кластеров, поэтому нам как раз помогает п.1)
# 3. Далее мы считаем, что клетки, которые алгоритм отнес к одному кластеру являются одним клеточным типом (если кластеров больше, чем типов клеток, то возможно, что 2 разных кластера являются одним типом)
# 4. После чего по интенсивности экспрессии поверхностных маркеров мы присваиваем кластеру клеточный тип

# Давайте для начала напишем удобную функцию для визуализации наших многомерных данных в пространстве размерностью 2, делать мы это будем при помощи алгоритма t-SNE.

# Поиграться с красивой визуализацией можно [тут](https://distill.pub/2016/misread-tsne/).

# In[33]:


def plot_tsne(data, n_iter=1000, 
              perplexity=40, color=None):
    """
    Функция для отрисовки результатов работы t-SNE
        :param data: таблица объекты х признаки
        :param n_iter: число итераций градиентного спуска,
            может быть полезно увеличить, чтобы получить результаты получше
        :param perplexity: 
        :param color: np.array[n_samples] с переменной,
            которой мы хотим покрасить наши наблюдения
        :return tsne_emb: np.array[n_samples, 2] - результаты работы t-SNE
    """
    
    # Сначала сделаем PCA, так как это хорошее начальное приближение для t-SNE
    # позволит алгоритму быстрее сойтись
    pca = PCA().fit(data)
    pca_embedding = pca.transform(data)
    
    # Запустим t-SNE, он выдаст нам для каждого объекта по 2 числа, 
    # которые мы сможем нарисовать
    tnse = TSNE(n_components=2, init=pca_embedding[:, :2], n_jobs=-1,
                n_iter=n_iter, perplexity=perplexity)
    tsne_embedding = tnse.fit_transform(pca_embedding)
    
    sns.scatterplot(x=tsne_embedding[:, 0],
                    y=tsne_embedding[:, 1],
                    hue=color, palette="colorblind")
    plt.xlabel("$TSNE_1$", fontsize=18)
    plt.ylabel("$TSNE_2$", fontsize=18)
    
    # Вернем также результаты t-SNE, так как, если потом захотим перестроить картинку,
    # в таком случае нам не придется ждать заново, просто нарисуем новую с готовыми данными
    
    return tsne_embedding


# In[34]:


tsne_res = plot_tsne(fc_data)


# Кластеризуйте ваши данные:
# 
# 1. Попробуйте методы кластеризации из тех, что мы прошли
# 2. Выберите лучший на основании метрики `silhouette_score` (попробуйте также подобрать гиперпараметры)
# 3. Присвойте каждому наблюдению метку класса и нарисуйте график t-SNE, покрасив точки метками кластера

# In[35]:


from sklearn.cluster import DBSCAN


# In[36]:


db = DBSCAN(eps=5).fit(tsne_res)
np.unique(db.labels_)


# In[37]:


print(f"Silhouette score is {silhouette_score(fc_data, db.labels_)}")


# In[38]:


sns.scatterplot(x = tsne_res[:, 0], y = tsne_res[:, 1],
                hue=db.labels_, palette="colorblind", marker="o");
plt.xlabel("$TSNE_1$", fontsize=18);
plt.ylabel("$TSNE_2$", fontsize=18);


# Удалось ли вам получить ситуацию, где отдельные группы точек покрашены в один цвет?

# ### ДА!

# #### 3.3. Аннотация клеточных типов
# 
# 4 балла

# Теперь когда мы подобрали хороший алгоритм кластеризации, можно аннотировать наши клетки. Для этого мы нарисуем t-SNE и покрасим точки в зависимости от интенсивности экспрессии поверхностных маркеров. В датасете присутствуют следующие типы клеток:
# 
# * B_cells
# * T_cells
# * Monocytes
# * Other cells
# 
# Вам нужно будет выяснить, какие клетки экспрессируют определенные маркеры и присвоить каждому кластеру один из типов клеток.

# Для начала нарисуем все это безобразие:

# In[39]:


# Результаты t-SNE уже есть в переменной tsne_res
fig, axes = plt.subplots(3, 3, figsize=(20, 20))
for col, ax in zip(fc_data.columns, axes.ravel()):
    scatter = ax.scatter(tsne_res[:, 0], tsne_res[:, 1], 
                         c=fc_data[col], cmap="YlOrBr")
    fig.colorbar(scatter, ax=ax)
    ax.set_title(col)
    ax.grid(False)
    ax.axis("off")


# Дальше дело за вами, нужно определить клеточный тип для каждого кластера и записать их как значения в словаре:

# In[40]:


cell_type_cluster_map = {0: "Other cells", 
                         1: "B_cells", 
                         2: "Other cells", 
                         3: "T_cells",
                         4: "Monocytes"}


# In[41]:


labeled_fc_data = fc_data.assign(Population=db.labels_)
labeled_fc_data["Population"] = labeled_fc_data["Population"].map(cell_type_cluster_map)


# Посчитайте, как распределены типы клеток:

# In[42]:


count = labeled_fc_data.groupby('Population').Population.count().sort_values()


# In[43]:


sns.barplot(x=count.index, y=count.values);


# Сохраните ваши результаты в csv файл, мы сравним их с правильными ответами по метрике `f1_score(average="macro")`, которая хорошо подходит, даже если классы не сбалансированы.

# In[44]:


labeled_fc_data.to_csv('data/labeled_fc_data')

