import numpy as np
import random
import copy
import pandas
from typing import NoReturn, Tuple, List,Union,Type

# Task 1
def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M), 
        0 --- злокачественной (B).

    
    """
    data = pandas.read_csv(f'{path_to_csv}').sample(frac= 1)
    y = data['label'].replace(['M','B'],['1','0']).astype(np.int8).to_numpy()
    X = data.drop(columns = ['label']).astype(np.float64).to_numpy()
    return X ,y


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """ 
    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток, 
        1 если сообщение содержит спам, 0 если не содержит.
    
    """
    data = pandas.read_csv(f'{path_to_csv}',index_col= False).sample(frac= 1)
    y = data['label'].to_numpy().astype(np.int64)
    X = data.drop(columns = ['label']).to_numpy().astype(np.float64)
    X_Maxabs = []
    X_minmax = []
    X_stdscaler = []
    for col in X.T:
        X_Maxabs.append(col/np.max(np.abs(col)))
        X_minmax.append ( (col - np.min(col) )/ (np.max(col) - np.min(col)))
        X_stdscaler.append( (col - np.mean(col) )/ np.std(col))

    X_Maxabs = np.array(X_Maxabs).T
    X_minmax = np.array(X_minmax).T
    X_stdscaler = np.array(X_stdscaler).T
    return X_Maxabs,y


# Task 2

def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    n = len(y)
    mapp = int(ratio * n)
    X_train = X[:mapp]
    y_train = y[:mapp]
    X_test = X[mapp:]
    y_test = y[mapp:]
    return X_train ,y_train ,X_test,y_test
    
# Task 3
def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
    classes = np.unique(y_true)
    n = len(y_pred)
    m = len(classes)

    recall  =  np.zeros(m)
    precision = np.zeros(m)

    l = 0
    for positive in classes:
        tp = 0
        tn = 0
        fn = 0
        fp = 0
        
        for i in range(n):
            if y_true[i] == positive and y_pred[i] == positive:
                tp += 1
            if y_true[i] != positive and y_pred[i] == positive:
                fp += 1
            if  y_true[i] != positive and  y_pred[i] != positive:
                tn += 1
            if  y_true[i] == positive and y_pred[i] != positive:
                fn += 1

        recall[l] = tp / (tp + fn)
        precision[l] = tp/(tp + fp)
        l += 1

    correct_predictions = 0
    for yt, yp in zip(y_true, y_pred):
        
        if yt == yp:
            
            correct_predictions += 1
    

    return precision,recall,correct_predictions / len(y_true)
                

    
# Task 4

class KDTree:

    def __init__(self, X: np.array, leaf_size: int = 30):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области, 
            в которых не меньше leaf_size точек).

        Returns
        -------

        """        
        self.X = np.hstack([np.arange(X.shape[0]).reshape(-1, 1), X])
        self.dim = X.shape[1]
        self.leaf_size = leaf_size
        self.root = self.tree(self.X, 0)

    class Node:

        def __init__(self,points: Union[None,np.array] = None, axis: Union[None,np.float64] = None, median:  Union[None,np.float64] = None):
            """
            Parameters:
            ----------
            points: Union[None,np.array]
                    Сюда заходят точки в конце разбиения, в остальном случае принмиает None
            axis :  Union[None,np.float] 
                    Сюда входят номера 'осей' признакового пространства, по которому оно разбивается
            median: Union[None,np.float]
                    Сюда заходят медианы

            Returns
            ----------
            """
            self.points = points
            self.axis = axis
            self.median = median
            self.l = None
            self.r = None        

    @staticmethod
    def metrics(a: np. array,b: np.array,ax : int = 1) -> np.array:
      
      """
      Parameters:
      ----------
      a, b : np.array 
            Точкa a (Точки a ) между которой (которыми) считается расстояние до точки b
      ax : int
            Размерность по которой будет считаться расстояния
      

      Returns: 
      np.array
      ----------
      """
      if b.size == 0 or a.size == 0:
          b = np.array( [b])
          a =np.array( [a] )
      return np.linalg.norm((a - b), axis=ax)

    @staticmethod
    def merge(opposInd: np.array , opposDist: np.array , indexes: np.array , neighDist: np.array ,k : int) -> Tuple[np.array,np.array]:
        
        """
        Parameters:
        ----------
        opposInd, indexes : np.array 
            opposInd, indexes -векторы индексов обходимых точек
        opposDist,neighDist : np.array
            opposDist,neighDist -векторы расстояний от выбираемой точки до соседних
            
        Returns:
        Tuple[np.array,np.array]
        ----------
        """
        n = m = 0
        mergedIdx = []
        mergedDist = []

        while n < len(opposInd) and m < len(indexes) and n + m < k:
            if opposDist[n] <= neighDist[m]:
                mergedIdx.append(opposInd[n])
                mergedDist.append(opposDist[n])
                n += 1
            else:
                mergedIdx.append(indexes[m])
                mergedDist.append(neighDist[m])
                m += 1


        mergedIdx.extend(opposInd[n: k - m])
        mergedDist.extend(opposDist[n: k - m])
        mergedIdx.extend(indexes[m: k - n])
        mergedDist.extend(neighDist[m: k - n])
      
        return mergedDist, mergedIdx
      

    def tree(self,X: np.array ,depth: int = 0) -> Union[None,Type[Node]]:
        
        """
        Parameters:
        ----------
        X: np.array
            Точки по которым будет строится дерево рекурсивно

        depth: int
            Глубина дерева
        
        Returns:
            None или Node
        ----------
        """
        axis = (depth % self.dim) + 1
        median = np.median(X[:,axis])
      
        lpart,rpart = X[ X[:,axis] < median], X[ X[:,axis] >= median]

        if lpart.shape[0] < self.leaf_size and rpart.shape[0] < self.leaf_size:
            return self.Node(points = X)
      
        root = self.Node(axis = axis,median = median)

        root.l = self.tree(lpart,depth + 1)
        root.r = self.tree(rpart,depth + 1)
        return root

    def closest(self, root: Node, point: np.array, k: int) -> Tuple[np.array,np.array]:
        """
        Parameters:
        ----------
        root: Node
            Листья
        point: np.array
            Точка от которой будет обходиться дерево
        k: int
            Количество ближайших соседей
        
        Returns: Tuple[np.array,np.array]
            Расстояние до ближайших соседей и их индексы
        ----------   
        """
        if root.l is None and root.r is None:

            neigh_dist = self.metrics(root.points[:, 1:], point, ax=1)
            neigh_index = np.argsort(neigh_dist)
            if root.points.size == 0:
                neigh_dist = np.array([0]*k)
                index = np.array([0]*k)
                return neigh_dist,index
            else:
                index = root.points[neigh_index][:k]
                return neigh_dist[neigh_index], index

        else:
            axis = root.axis -1
            if root.median > point[axis]:
                neigh_dist, index = self.closest(root.l, point, k)
                opposite = root.r

            else:
                neigh_dist, index = self.closest(root.r, point, k)
                opposite = root.l
            
            if neigh_dist[-1] >= self.metrics(point[axis], root.median,ax = None) or len(index) < k:
                    opposite_dist, opposite_idx = self.closest(opposite, point, k)
                    return self.merge(opposite_idx, opposite_dist, index, neigh_dist,k)

            return neigh_dist, index           
    
    def query(self, X: np.array, k: int = 1) -> List[List]:
        """

        Parameters:
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.

        Returns:
        ----------
        list[list]
            Список списков (длина каждого списка k): 
            индексы k ближайших соседей для всех точек из X.

        """
        
        res = []
        for x in X:
          point_neigh = self.closest(self.root,point = x, k =k)[1]
          point_neigh = np.array(point_neigh)

          try:
              res.append(point_neigh[:,0].astype(dtype = np.int64).tolist())
          except:
              #res.append(point_neigh.tolist())
              res.append([1]*k)
        return res
        
# Task 5

class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """        
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        
    
    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """        

        self.X = X
        self.y = y
        self.kdtree = KDTree(X,self.leaf_size) 
        self.labels = np.unique(np.sort(y))

    
    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.
            

        """
    
        predict_k_labels = self.y[self.kdtree.query(X, k=self.n_neighbors)]

        proba = np.zeros((X.shape[0], self.labels.shape[0]))
        for i, c in enumerate(self.labels):
            proba[:, i] += (predict_k_labels == c).sum(1)
        proba /= self.n_neighbors
        return proba
     
    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        np.array
            Вектор предсказанных классов.
            

        """
        return np.argmax(self.predict_proba(X), axis=1)
