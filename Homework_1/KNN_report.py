#!/usr/bin/env python
# coding: utf-8

# # Домашнее задание №1 - Метод К-ближайших соседей (K-neariest neighbors)
# 
# Сегодня мы с вами реализуем наш первый алгоритм машинного обучения, метод К-ближайших соседей. Мы попытаемся решить с помощью него задачи:
# - бинарной классификации (то есть, только двум классам)
# - многоклассовой классификации (то есть, нескольким классам)
# - регрессии (когда зависимая переменная - натуральное число)
# 
# Так как методу необходим гиперпараметр (hyperparameter) - количество соседей, то нам нужно научиться подбирать этот параметр. Мы постараемся научиться пользовать numpy для векторизованных вычислений, а также посмотрим на несколько метрик, которые используются в задачах классификации и регрессии.
# 
# Перед выполнением задания:
# - установите все необходимые библиотеки, запустив `pip install -r requirements.txt`
# 
# Если вы раньше не работали с numpy или позабыли его, то можно вспомнить здесь:  
# http://cs231n.github.io/python-numpy-tutorial/

# In[1]:


import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import pandas as pd


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from knn import KNNClassifier
from metrics import binary_classification_metrics, multiclass_accuracy



# In[2]:


# plt.rcParams["figure.figsize"] = 12, 9
sns.set_style("whitegrid")

SEED = 111
random.seed(SEED)
np.random.seed(SEED)


# In[3]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Задание 1. KNN на датасете Fashion-MNIST (10 баллов)

# В этом задании вам предстоит поработать с картинками одежды, среди которых можно выделить 10 классов. Данные уже загружены за вас: в переменной X лежат 70000 картинок размером 28 на 28 пикселей, вытянутые в вектор размерностью 784 (28 * 28). Так как данных довольно много, а наш KNN будет весьма медленный, то возьмем случайно 1000 наблюдений (в реальности в зависимости от вашей реализации можно будет взять больше, но если будет не зватать ОЗУ, то берите меньше).

# In[4]:


X, y = fetch_openml(name="Fashion-MNIST", return_X_y=True, as_frame=False)


# In[5]:


idx_to_stay = np.random.choice(np.arange(X.shape[0]), replace=False, size=1000)
X = X[idx_to_stay]
y = y[idx_to_stay]


# Давайте посмотрим на какое-нибудь изображение из наших данных:

# In[6]:


# возьмем случайную картинку и сделаем reshape
# 28, 28, 1 = H, W, C (число каналов, в данном случае 1)
image = X[np.random.choice(np.arange(X.shape[0]))].reshape(28, 28, 1)
plt.imshow(image)
plt.axis("off");


# ### 1.1. Посмотрим на все классы (0.5 баллов)

# Возьмите по одной картинке каждого класса и изобразите их (например, сделайте subplots 5 на 2).

# In[7]:


classes = np.unique(y)


# In[8]:


Y_series = pd.Series(y)

indexes = []
for i in range(10):
    indexes.append(Y_series.loc[Y_series == str(i)].index[0])    


# In[9]:


images = []
for i in indexes:
    images.append(X[i])

fig, axs = plt.subplots(1, 10, figsize=(15, 5))

for i in range(0, 10):
    axs[i].imshow(images[i].reshape(28,28,1))
    axs[i].axis('off')


# ### 1.2. Сделайте небольшой EDA (1 балл)

# Посмотрите на баланс классов. В дальнейших домашках делайте EDA, когда считаете нужным, он нужен почти всегда, но оцениваться это уже не будет, если не будет указано иное. Делайте EDA, чтобы узнать что-то новое о данных!

# In[14]:


pd.Series(y).describe()


# In[15]:


fig, axes = plt.subplots(figsize=(6, 4), tight_layout=True);
sns.histplot(sorted(y));
plt.xlabel('class');


# ### 1.3. Разделите данные на train и test (0.5 баллов)

# Разделите данные на тренировочную и тестовую выборки, размеры тестовой выборки выберите сами. Здесь вам может помочь функция `train_test_split`

# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ### 1.4. KNN для бинарной классификации (6 баллов)

# Давайте возьмем для задачи бинарной классификации только объекты с метками классов 0 и 1.

# In[17]:


binary_train_y = y_train[(y_train == '1') | (y_train == '0')]
binary_train_X = X_train[:binary_train_y.shape[0]]

binary_test_y = y_test[(y_test == '1') | (y_test == '0')]
binary_test_X = X_test[:binary_test_y.shape[0]]


# И вот мы подготовили данные, но модели у нас пока что нет. В нескольких занятиях нашего курса вам придется самостоятельно реализовывать какие-то алгоритмы машинного обучения, а потом сравнивать их с готовыми библиотечными решениями. В остальных заданиях реализовывать алгоритмы будет не обязательно, но может быть полезно, поэтому часто это будут задания на дополнительные баллы, но главное не это, а понимание работы алгоритма после его реализации с нуля на простом numpy. Также это все потом можно оформить в виде репозитория ml_from_scratch и хвастаться перед друзьями.

# In[18]:


knn_classifier = KNNClassifier(k=1)
knn_classifier.fit(binary_train_X, binary_train_y)


# ### Настало время писать код!

# В KNN нам нужно для каждого тестового примера найти расстояния до всех точек обучающей выборки. Допустим у нас 1000 примеров в train'е и 100 в test'е, тогда в итоге мы бы хотели получить матрицу попарных расстояний (например, размерностью 100 на 1000). Это можно сделать несколькими способами, и кому-то наверняка, в голову приходит идея с двумя вложенными циклами (надеюсь, что не больше:). Так можно делать, то можно и эффективнее. Вообще, в реальном KNN используется структура данных [k-d-tree](https://ru.wikipedia.org/wiki/K-d-%D0%B4%D0%B5%D1%80%D0%B5%D0%B2%D0%BE), которая позволяет производить поиск за log(N), а не за N, как будем делать мы (по сути это такое расширение бинарного поиска на многомерное пространство).
# 
# Вам нужно будет последовательно реализовать методы `compute_distances_two_loops`, `compute_distances_one_loop` и `compute_distances_no_loops` класса `KNN` в файле `knn.py`.
# 
# Эти функции строят массив расстояний между всеми векторами в тестовом наборе и в тренировочном наборе.  
# В результате они должны построить массив размера `(num_test, num_train)`, где координата `[i][j]` соотвествует расстоянию между i-м вектором в test (`test[i]`) и j-м вектором в train (`train[j]`).
# 
# **Обратите внимание** Для простоты реализации мы будем использовать в качестве расстояния меру L1 (ее еще называют [Manhattan distance](https://ru.wikipedia.org/wiki/%D0%A0%D0%B0%D1%81%D1%81%D1%82%D0%BE%D1%8F%D0%BD%D0%B8%D0%B5_%D0%B3%D0%BE%D1%80%D0%BE%D0%B4%D1%81%D0%BA%D0%B8%D1%85_%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B0%D0%BB%D0%BE%D0%B2)).
# 
# $d_{1}(\mathbf {p} ,\mathbf {q} )=\|\mathbf {p} -\mathbf {q} \|_{1}=\sum _{i=1}^{n}|p_{i}-q_{i}|$

# В начале я буду иногда писать разные assert'ы, чтобы можно было проверить правильность реализации, в дальнейшем вам нужно будет их писать самим, если нужно будет проверять корректность каких-то вычислений.

# In[19]:


dists = knn_classifier.compute_distances_two_loops(binary_test_X)
assert np.isclose(dists[0, 100], np.sum(np.abs(binary_test_X[0] - binary_train_X[100])))


# In[20]:


dists = knn_classifier.compute_distances_one_loop(binary_test_X)
assert np.isclose(dists[0, 100], np.sum(np.abs(binary_test_X[0] - binary_train_X[100])))


# In[21]:


dists = knn_classifier.compute_distances_no_loops(binary_test_X)
assert np.isclose(dists[0, 100], np.sum(np.abs(binary_test_X[0] - binary_train_X[100])))


# Проверим скорость работы реализованных методов

# In[22]:


get_ipython().run_line_magic('timeit', 'knn_classifier.compute_distances_two_loops(binary_test_X)')
get_ipython().run_line_magic('timeit', 'knn_classifier.compute_distances_one_loop(binary_test_X)')
get_ipython().run_line_magic('timeit', 'knn_classifier.compute_distances_no_loops(binary_test_X)')


# Реализуем метод для предсказания меток класса

# In[23]:


prediction = knn_classifier.predict(binary_test_X)
prediction


# ### Метрика

# Теперь нужно реализовать несколько метрик для бинарной классификации. Не забудьте подумать о численной нестабильности (деление на 0).

# In[24]:


binary_classification_metrics(prediction, binary_test_y.astype('int64'))


# Все ли хорошо с моделью? Можно проверить свою реализацию с функциями из библиотеки `sklearn`:

# <img src="https://i.imgflip.com/406fu9.jpg" width="800" height="400">

# In[25]:


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# In[26]:


accuracy_score(binary_test_y.astype('int64'), prediction)


# In[27]:


precision_score(binary_test_y.astype('int64'), prediction)


# In[28]:


recall_score(binary_test_y.astype('int64'), prediction)


# In[30]:


f1_score(binary_test_y.astype('int64'), prediction)


# СОВПАДАЕТ!

# ### Подбор оптимального k

# Чтобы подрбрать оптимальное значение параметра k можно сделать следующее: задать область допустимых значений k, например, `[1, 3, 5, 10]`. Дальше для каждого k обучить модель на тренировочных данных, сделать предсказания на тестовых и посчитать какую-нибудь метрику (метрику выберите сами исходя из задачи, но постарайтесь обосновать выбор). В конце нужно посмотреть на зависимость метрики на train'е и test'е от k и выбрать подходящее значение.
# 
# Реализуйте функцию `choose_best_k` прямо в ноутбуке.

# In[49]:


def find_best_k(X_train, y_train, X_test, y_test, params, metric):
    """
    Choose the best k for KKNClassifier
    Arguments:
    X_train, np array (num_train_samples, num_features) - train data
    y_train, np array (num_train_samples) - train labels
    X_test, np array (num_test_samples, num_features) - test data
    y_test, np array (num_test_samples) - test labels
    params, list of hyperparameters for KNN, here it is list of k values
    metric, function for metric calculation
    Returns:
    train_metrics the list of metric values on train data set for each k in params
    test_metrics the list of metric values on test data set for each k in params
    """
    train_metrics, test_metrics = [], []
    
    for k in params:
        knn_classifier = KNNClassifier(k=k)
        knn_classifier.fit(X_train, y_train)
        
        y_pred_train = knn_classifier.predict(X_train)
        y_pred_test = knn_classifier.predict(X_test)
        
        train_metrics.append(metric(y_train.astype('int64'), y_pred_train.astype('int64')))
        test_metrics.append(metric(y_test.astype('int64'), y_pred_test.astype('int64')))
    
    return train_metrics, test_metrics
        


# **Я выбрал метрику accuracy, потому что здесь нет "позитивных и отрицательных" исходов, здесь 2 равнозначных класса**  
# То есть True Negative и True Positive имеют одинаковый вклад

# In[50]:


params = [1, 2, 4, 5, 8, 10, 15, 20, 25, 30, 50]
train_metrics, test_metrics = find_best_k(binary_train_X, binary_train_y, binary_test_X, binary_test_y, params, accuracy_score)


# In[51]:


plt.plot(params, train_metrics, label="train")
plt.plot(params, test_metrics, label="test")
plt.legend()
plt.xlabel("K in KNN")
plt.ylabel("ACCURACY");


# **k = 5 - вполне нормально**

# На самом деле, это не самый лучший способ подбирать гиперпараметры, но способы получше мы рассмотрим в следующий раз, а пока что выберите оптимальное значение k, сделайте предсказания и посмотрите, насколько хорошо ваша модель предсказывает каждый из классов.

# ### 1.5. Многоклассоввая классификация (2 балла)

# Теперь нужно научиться предсказывать все 10 классов. Для этого в начале напишем соответствующий метод у нашего классификатора.

# In[52]:


knn_classifier = KNNClassifier(k=2)
knn_classifier.fit(X_train, y_train)


# In[53]:


predictions = knn_classifier.predict(X_test)
predictions


# Осталось реализовать метрику качества для многоклассовой классификации, для этого реализуйте функцию `multiclass_accuracy` в `metrics.py`.

# In[43]:


multiclass_accuracy(predictions, y_test)


# Снова выберите оптимальное значение K как мы делали для бинарной классификации.

# In[54]:


params = [1, 2, 4, 5, 8, 10, 20, 30]
train_metrics, test_metrics = find_best_k(X_train, y_train, X_test, y_test, params, accuracy_score)


# In[55]:


plt.plot(params, train_metrics, label="train")
plt.plot(params, test_metrics, label="test")
plt.legend()
plt.xlabel("K in KNN")
plt.ylabel("ACCURACY");


# **k= 5 - вполне нормально**

# ## Задание 2. KNN на датасете diabetes (10 баллов)

# Теперь попробуем применить KNN к задаче регрессии. Будем работать с [данными](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) о диабете. В этом задании будем использовать класс `KNeighborsRegressor` из библиотеки `sklearn`. Загрузим необходимые библиотеки:

# In[56]:


from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

from metrics import r_squared, mse, mae


# In[57]:


X, y = load_diabetes(as_frame=True, return_X_y=True)


# In[58]:


new_colnames = dict(zip(X.columns[4:],["tc", "ldl", "hdl","tch", "ltg", "glu"]))
X = X.rename(columns=new_colnames)
X.head()


# ### 2.1. EDA (2 обязательных балла + 2 доп. балла за Pipeline)

# Сделайте EDA, предобработайте данные так, как считаете нужным, нужна ли в данном случае стандартизация и почему? Не забудте, что если вы стандартизуете данные, то нужно считать среднее и сдандартное отклонение на тренировочной части и с помощью них трансформировать и train, и test (**если не поняли это предложение, то обязательно разберитесь**).
# 
# **Дополнительно**:
# Попробуйте разобраться с [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), чтобы можно было создать класс, который сразу проводит стандартизацию и обучает модель (или делает предсказание). Пайплайны очень удобны, когда нужно применять различные методы предобработки данных (в том числе и к разным столбцам), а также они позволяют правильно интегрировать предобработку данных в различные классы для поиска наилучших гиперпараметров модели (например, `GridSearchCV`).

# In[59]:


from sklearn.pipeline import Pipeline


# In[60]:


X.info()


# ### Мультиколинеарность

# In[61]:


plt.rcParams['figure.figsize'] = (8,6)
sns.heatmap(X.corr(), annot=True, fmt='.2f');


# **tc, hdl, ldl, tch сильно коррелированы друг с другом, так как обозначают схожые параметры**

# ### Распределение целевой переменной

# In[62]:


sns.histplot(y);


# В описании данных написано:  
# **Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times the square root of n_samples (i.e. the sum of squares of each column totals 1).**  
# Поэтому стандартизация уже не нужна
# 
# НО! Я все равно сделаю, чтобы потренироваться в пайплайнах))
# 
# ### Пусть k=5

# In[63]:


knn_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=5))
])


# ### 2.2. Регрессионная модель (1 балл)

# Создайте модель `KNeighborsRegressor`, обучите ее на треноровочных данных и сделайте предсказания.

# In[64]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[65]:


# knn_regressor = KNeighborsRegressor(n_neighbors=5);
# knn_regressor.fit(X_train, y_train);
knn_pipeline.fit(X_train, y_train);


# In[66]:


y_pred = knn_pipeline.predict(X_test)
# y_pred = knn_regressor.predict(X_test)


# ### 2.3. Метрики регресии (3 балла)

# Реализуйте метрики $R^2$, MSE и  MAE в `metrics.py`. Примените их для оценки качества полученной модели. Все ли хорошо?
# 
# Напомню, что:
# 
# $R^2 = 1 - \frac{\sum_i^n{(y_i - \hat{y_i})^2}}{\sum_i^n{(y_i - \overline{y})^2}}$
# 
# $MSE = \frac{1}{n}\sum_i^n{(y_i - \hat{y_i})^2}$
# 
# $MAE = \frac{1}{n}\sum_i^n{|y_i - \hat{y_i}|}$

# ### Для k = 5

# In[67]:


r_squared(y_pred, y_test)


# In[68]:


r2_score(y_test, y_pred)


# In[69]:


mse(y_pred, y_test)


# In[70]:


mean_squared_error(y_test, y_pred)


# In[71]:


mae(y_pred, y_test)


# In[72]:


mean_absolute_error(y_test, y_pred)


# ### 2.4. Подбор оптимального числа соседей (2 балла)

# Мы почти дошли до конца. Теперь осталось при помощи реализованных нами метрик выбрать лучшее количество соседей для нашей модели.
# 
# !!! Обратите внимание на то, что значат наши метрики, для некоторых хорошо, когда они уменьшаются, для других наоборот.

# In[73]:


from metrics import r_squared, mse, mae


# In[74]:


def find_best_k(X_train, y_train, X_test, y_test, params, r2=r_squared, mae=mae, mse=mse):
    """
    Choose the best k for KNeighborsRegressor
    
    Arguments:
    X_train, np array (num_train_samples, num_features) - train data
    y_train, np array (num_train_samples) - train labels
    X_test, np array (num_test_samples, num_features) - test data
    y_test, np array (num_test_samples) - test labels
    params, list of hyperparameters for KNN, here it is list of k values
    r2, function calculates the R squared value, r2(y_pred, y_test)
    mse, function calculates the Mean squared error value, mse(y_pred, y_test)
    mae, function calculates the Mean absolute error value, mae(y_pred, y_test)
    
    Returns:
    None
    """
    train_r2, test_r2, train_mse, test_mse, train_mae, test_mae = [], [], [], [], [], []
    
    for k in params:
        knn_pipeline = Pipeline(steps=[("scaler", StandardScaler()),
                                       ("knn", KNeighborsRegressor(n_neighbors=k))])
        
        knn_pipeline.fit(X_train, y_train)
        
        y_pred_train = knn_pipeline.predict(X_train)
        y_pred_test = knn_pipeline.predict(X_test)
        
        r_sq_train = r2(y_pred_train, y_train)
        r_sq_test = r2(y_pred_test, y_test)
        
        mse_train = mse(y_pred_train, y_train)
        mse_test = mse(y_pred_test, y_test)
        
        mae_train = mae(y_pred_train, y_train)
        mae_test = mae(y_pred_test, y_test)
        
        print(
            f''' 
            <------- for k = {k} ------>:
            
            R2 for train data = {r_sq_train}
            R2 for test data = {r_sq_test}
            
            MSE for train data = {mse_train}
            MSE for test data = {mse_test}
            
            MAE for train data = {mae_train}
            MAE for test data = {mae_test}'''
             )
        
        train_r2.append(r_sq_train)
        test_r2.append(r_sq_test)
        
        train_mse.append(mse_train)
        test_mse.append(mse_test)
        
        train_mae.append(mae_train)
        test_mae.append(mae_test)
    
    return train_r2, test_r2, train_mse, test_mse, train_mae, test_mae


# Для поиска лучшего k вы можете воспользоваться функцией `find_best_k`, которую вы реализовали выше.

# In[75]:


params = [1, 2, 4, 5, 8, 10, 30]
train_r2, test_r2, train_mse, test_mse, train_mae, test_mae = find_best_k(X_train, y_train, X_test, y_test, params)


# In[76]:


plt.plot(params, train_r2, label="train")
plt.plot(params, test_r2, label="test")
plt.legend()
plt.xlabel("K in KNN")
plt.ylabel("R2");
plt.title('R Squared');


# In[77]:


plt.plot(params, train_mse, label="train")
plt.plot(params, test_mse, label="test")
plt.legend()
plt.xlabel("K in KNN")
plt.ylabel("MSE");
plt.title('MSE');


# In[78]:


plt.plot(params, train_mae, label="train")
plt.plot(params, test_mae, label="test")
plt.legend()
plt.xlabel("K in KNN")
plt.ylabel("MAE");
plt.title('MAE');


#  ### Кажется, значение k = 10 будет вполне приемлимо

# ### 3. Социализация (0.5 доп. балла)
# 
# Так как у нас теперь большая группа, то было бы здорово всем познакомиться получше (так как выпускной не за горами). Соберитесь с одногруппниками в зуме (желательно, чтобы были люди и с Онлайна, и с Питера), познакомьтесь, а сюда прикрепите скриншот с камерами всех участников.

# ## Therapy time

# Напишите здесь ваши впечатления о задании: было ли интересно, было ли слишком легко или наоборот сложно и тд. Также сюда можно написать свои идеи по улучшению заданий, а также предложить данные, на основе которых вы бы хотели построить следующие дз. 

# **Ваши мысли:**

# # Спасибо, что отвечаете на вопросы!
