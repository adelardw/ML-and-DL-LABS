from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List

# Task 1


def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    values = np.unique(x)
    n = len(x)
    sum = 0

    for val in values:
        p = len(x[x == val]) / n
        sum += p * (1 - p)

    return sum


def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """

    values = np.unique(x)
    n = len(x)
    sum = 0

    for val in values:
        p = len(x[x == val]) / n
        sum += p * np.log2(p)

    return -sum


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    node = np.hstack((left_y, right_y))
    return criterion(node) - criterion(left_y)*(left_y.size/node.size) - (
           criterion(right_y)*(right_y.size/node.size))



# Task 2


class DecisionTreeLeaf:
    """
    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа
        дерева
    """
    def __init__(self, ys):
        unique, value_counts = np.unique(ys, return_counts=True)
        self.prob = dict(zip(unique, value_counts/ys.size))
        self.y = unique[np.argmax(value_counts)]


class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value. 
    """
    def __init__(self, split_dim: int, split_value: float, 
                 left: Union['DecisionTreeNode', DecisionTreeLeaf], 
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
        
# Task 3

class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """
    def __init__(self, criterion: str = "gini",
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = gini if criterion == 'gini' else entropy
        self.root = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """
        self.n_samples, self.n_features = X.shape
        self.y_unique = np.unique(y)

        def btree(X, y, depth=0):
            depth +=1
            if len(y) > self.min_samples_leaf:
                max_gain = -1e10
                for i in range(self.n_features):
                    gain, border_i = search_Xborder(X[:, i], y)
                    if gain > max_gain:
                        max_gain = gain
                        border = border_i
                        feature_on_max_gain = i

                left_x = X[X[:, feature_on_max_gain] < border]
                right_x = X[X[:, feature_on_max_gain] >= border]
                left_y = y[X[:, feature_on_max_gain] < border]
                right_y = y[X[:, feature_on_max_gain] >= border]

                if self.max_depth is not None:
                    if len(left_x) == 0 or len(right_x) == 0 or depth > self.max_depth:
                        return DecisionTreeLeaf(y)
                else:
                    if len(left_x) == 0 or len(right_x) == 0:
                        return DecisionTreeLeaf(y)

                left = btree(left_x, left_y, depth)
                right = btree(right_x, right_y, depth)

                return DecisionTreeNode(split_dim=feature_on_max_gain,
                                        split_value=border,
                                        left=left,
                                        right=right)
            return DecisionTreeLeaf(y)
            
        def search_Xborder(X, y):
            max_i_gain = -1e10
            uniq_values = np.unique(X)
            for i, split_val in enumerate(uniq_values):
                left_y, right_y = y[X < split_val], y[X >= split_val]
                information_gain = gain(left_y, right_y, self.criterion)
                if information_gain > max_i_gain:
                    max_i_gain = information_gain
                    num_of_feature = i
            
            return max_i_gain, uniq_values[num_of_feature]

        
        self.root = btree(X, y, depth=0)

    
    def predict_proba(self, X: np.ndarray) ->  List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.
        
        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь 
            {метка класса -> вероятность класса}.
        """
        self.predicted = [dict.fromkeys(self.y_unique)]*X.shape[0]

        def search(node, X, index):
            if type(node) is DecisionTreeLeaf:
                for i in index:
                    self.predicted[i] = node.prob
                return self.predicted
            mask = X[:, node.split_dim]<node.split_value
            left_index, right_index = index[mask], index[~mask]
            left_find, right_find = X[mask], X[~mask]
            search(node.left, left_find, left_index)
            search(node.right, right_find, right_index)
            
        search(self.root, X, np.arange(X.shape[0]))
        return self.predicted

    
    def predict(self, X : np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.
        
        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]
    
# Task 4
task4_dtc = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30)

