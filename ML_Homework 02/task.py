from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons
from scipy.spatial import distance_matrix
import numpy as np
import random
import copy
import cv2
from collections import deque
from collections import defaultdict
from typing import NoReturn

# Task 1

class KMeans:
    def __init__(self, n_clusters: int, init: str = "random", 
                 max_iter: int = 300):
        """
        
        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из  X,
            3. k-means++ --- центроиды кластеров инициализируются 
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.
        
        """
        
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None

    @staticmethod
    def metrics(a,b):
      return np.linalg.norm(a - b)
    
    @staticmethod
    def random(n_cluste,feat_dim):
        np.random.seed(2)           
        centroids = np.random.rand(n_cluste,feat_dim)
        return centroids

    @staticmethod
    def sample(X,n_cluste):
        np.random.seed(2)           
        centroids_idx = np.random.randint(0,X.shape[0],n_cluste)
        centroids = X[centroids_idx,:]
        return centroids
    
    def kamin_plus_special_offer(self,X,n_clusters):
            
            centroids_idx = np.random.randint(0,X.shape[0],1)
            centroids = X[centroids_idx,:].tolist()
            k = 0
            while k < self.n_clusters - 1:
                dist = []
                for x in X:
                  d = np.inf
                  for j in range(len(centroids)):
                    temp_dist = self.metrics(x,centroids[j])
                    d = min(d, temp_dist)
                  dist.append(d)
                dist = np.array(dist)
                next_centroid = X[np.argmax(dist), :]
                centroids.append(next_centroid)
                k+=1
            return centroids
        
    def fit(self, X: np.array, y = None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.
        
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать 
            параметры X и y, даже если y не используется).
        
        """
        n_points = X.shape[0]
        feat_dim = X.shape[1]

        if self.init == "random":
            self.centroids = self.random(self.n_clusters,feat_dim) 
        if self.init =="sample":
            self.centroids = self.sample(X,self.n_clusters)
        if self.init =="k-means++":
           self.centroids = self.kamin_plus_special_offer(X,self.n_clusters)


    
    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера, 
        к которому относится данный элемент.
        
        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.
        
        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров 
            (по одному индексу для каждого элемента из X).
        
        """
        clusters = []

        for l in range(self.max_iter):
          clusters = [[] for i in range(self.n_clusters)]
          res = []
          for x in X:
            distance = [self.metrics(x, centr) for centr in self.centroids ]
            mindist = np.argmin(distance)
            clusters[mindist].append(x)
            res.append(mindist)
          self.centroids = [ np.mean(x,axis = 0) for x in clusters]

          if l == self.max_iter - 1:

              return np.array(res)
    
# Task 2

class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, 
                 leaf_size: int = 40, metric: str = "euclidean"):
        """
        
        Parameters
        ----------
        eps : float, min_samples : int
            Параметры для определения core samples.
            Core samples --- элементы, у которых в eps-окрестности есть 
            хотя бы min_samples других точек.
        metric : str
            Метрика, используемая для вычисления расстояния между двумя точками.
            Один из трех вариантов:
            1. euclidean 
            2. manhattan
            3. chebyshev
        leaf_size : int
            Минимальный размер листа для KDTree.

        """
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size= leaf_size
        self.metric = metric

    def dfs(self,v,components,graph,num_components):
      components[v] = num_components
      for u in graph[v]:
        if components[u] == -1:
          self.dfs(u,components,graph,num_components)

    def dist(self,X, intersection, idx):
        if self.metric == 'euclidean':
            dist = np.linalg.norm(X[intersection] - X[idx], axis=1)
        if self.metric == 'manhattan':
            dist = np.linalg.norm(X[intersection] - X[idx], axis=1, ord=1)
        if self.metric == 'chebyshev':
            dist = np.linalg.norm(X[intersection] - X[idx], axis=1, ord=np.inf)
        return dist
    
    def fit_predict(self, X: np.array, y = None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        tree = KDTree(X,leaf_size= self.leaf_size,metric = self.metric)
        neighbours = tree.query_radius(X,r = self.eps)
        n = X.shape[0]

        indexes = set(np.arange(n,dtype = int)[[len(x) >= self.min_samples for x in neighbours]])
        labels = -np.ones(n,dtype = int)
        num_components = 0


        adj = defaultdict(set)
        for x in indexes:
          current_neigh = neighbours[x]
          for u in current_neigh:
            if u in indexes:
              adj[x].add(u)
              adj[u].add(x)


        for x in indexes:
          if labels[x] == -1:
            self.dfs(x,labels,adj,num_components)
            num_components += 1

        for i, label in enumerate(labels):
              intersection = list(indexes.intersection(set(neighbours[i])))
              if label == -1 and len(intersection) !=0:

                nearest_neigh = self.dist(X,intersection,i).argsort()[0]
                labels[i] = labels[intersection[nearest_neigh]]

        return labels

# Task 3
def average(X: np.array, Y: np.array):
    return np.mean((X, Y), axis=0)

def single(X: np.array, Y: np.array):
    return np.minimum(X, Y)

def complete(X: np.array, Y: np.array):
    return np.maximum(X, Y)

class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """
        
        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры 
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        self.n_clusters=n_clusters
        if linkage == "average":
            self.linkage = average
        elif linkage == "single":
            self.linkage = single
        elif linkage == "complete":
            self.linkage = complete
        self.c_num=0
        self.cluster_list=[]


    
    def fit_predict(self, X: np.array, y = None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        self.c_num = X.shape[0]
        ans= np.full(self.c_num, -1)
        clusters = np.array([a for a in range(self.c_num)])
        cur = clusters

        dist = distance_matrix(X, X)
        np.fill_diagonal(dist, np.Inf)
      

        while self.n_clusters < cur.size :

            mini=np.argmin(dist)
            f_point = mini // self.c_num
            s_point = mini % self.c_num

            dist[f_point, :] = self.linkage(dist[f_point], dist[s_point])
            dist[:, f_point] = dist[f_point, :]

            dist[s_point, :] = np.Inf
            dist[:, s_point] = np.Inf
            dist[f_point, f_point] = np.Inf

            clusters[np.where(clusters == s_point)] = f_point
            cur = np.unique(clusters)

        num=0
        for i in cur:
            ans[np.where(clusters == i)] = num
            num += 1

        return ans