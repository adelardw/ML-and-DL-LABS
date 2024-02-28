import numpy as np
import copy
from cvxopt import spmatrix, matrix, solvers
from sklearn.datasets import make_classification, make_moons, make_blobs
from typing import NoReturn, Callable

solvers.options['show_progress'] = False

# Task 1


class LinearSVM:
    def __init__(self, C: float):
        """
        
        Parameters
        ----------
        C : float
            Soft margin coefficient.
        
        """
        self.C = C
        self.w = None
        self.b = None
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp
        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X 
            (можно считать, что равны -1 или 1). 
        
        """
        eps = 1e-5

        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        X = X.astype(float)
        y = y.astype(float)
        
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.dot(X[i], X[j])
        
        P = matrix(np.outer(y, y) * K)
        q = matrix(np.ones(n_samples) * -1)
        A = matrix(y.T, (1, n_samples))
        b = matrix(0.0)
        G = matrix(-np.eye(n_samples))
        h = matrix(np.zeros(n_samples))

        if self.C is None:
            G = matrix(-np.eye(n_samples))
            h = matrix(np.zeros(n_samples))
        else:

            G = matrix(np.vstack((-np.eye(n_samples),
                                  np.eye(n_samples))))
            h = matrix(np.hstack((np.zeros(n_samples),
                                  np.ones(n_samples) * self.C)))

        solution = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(solution['x'])
        y = y.reshape(-1, 1)
        self.w = ((y * alpha) * X).sum(axis=0)
        self.support = np.arange(alpha.shape[0])[alpha[:, 0] > eps]
        y = y[self.support]
        X = X[self.support]
        self.b = (y - X @ self.w.reshape(-1, 1))
        

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X 
            (т.е. то число, от которого берем знак с целью узнать класс).     
        """
        return (X @ self.w) + self.b.mean()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.   
        
        """
        return np.sign(self.decision_function(X))
    
    
def get_polynomial_kernel(c=1, power=2):
    def kernel(x, y):
        return (x @ y.T + c) ** power
    return kernel


def get_gaussian_kernel(sigma=1.):
    def kernel(x: np.ndarray, y: np.array):
        if x.ndim == 1:
            return np.exp(-sigma*np.linalg.norm(y - x)**2)
        else:
            return np.exp(-sigma*np.linalg.norm(y - x, axis=1)**2)
    return kernel

# Task 3
'''
class KernelSVM:
    def __init__(self, C: float, kernel: Callable):
        """
        
        Parameters
        ----------
        C : float
            Soft margin coefficient.
        kernel : Callable
            Функция ядра.
        
        """
        self.C = C
        self.kernel = kernel
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn: 
        """ 
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp 
         
        Parameters 
        ---------- 
        X : np.ndarray 
            Данные для обучения SVM. 
        y : np.ndarray 
            Бинарные метки классов для элементов X  
            (можно считать, что равны -1 или 1).  
         
        """ 
        n_samples = y.shape[0] 
        K = np.zeros((n_samples, n_samples)) 
        for i, x in enumerate(X): 
            K[i, :] = self.kernel(X, x) 
        P = matrix(np.outer(y, y) * K) 
        q = matrix(np.ones(n_samples) * -1) 
        A = matrix(y.astype('float'), size=(1, y.shape[0])) 
        b = matrix(0.0) 
        G = matrix(np.concatenate((-np.eye(n_samples), np.eye(n_samples)), axis=0)) 
        h = matrix(np.concatenate((np.zeros(n_samples), self.C * np.ones(n_samples)), axis=0)) 
 
        alpha = np.ravel(solvers.qp(P, q, G, h, A, b)['x']) 
 
        self.all_support = alpha > 1e-5 
        self.support = np.where((alpha > 1e-5) * (alpha < self.C - 1e-5))[0] 
        self.alpha_support = alpha[self.all_support] 
        self.X_all_support = X[self.all_support] 
        self.y_support = y[self.all_support] 
 
        self.b = np.sum(self.alpha_support * self.y_support * self.kernel(self.X_all_support, X[self.support[0]])) - y[self.support[0]] 
 
    def decision_function(self, X: np.ndarray) -> np.ndarray: 
        """ 
        Возвращает значение решающей функции. 
         
        Parameters 
        ---------- 
        X : np.ndarray 
            Данные, для которых нужно посчитать значение решающей функции. 
 
        Return 
        ------ 
        np.ndarray 
            Значение решающей функции для каждого элемента X  
            (т.е. то число, от которого берем знак с целью узнать класс).      
         
        """ 
        f = np.zeros(X.shape[0]) - self.b 
        for i, support in enumerate(self.X_all_support): 
            f += self.alpha_support[i] * self.y_support[i] * self.kernel(X, support) 
        return f

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.   
        
        """
        return np.sign(self.decision_function(X))
'''



class KernelSVM:
    def __init__(self, C: float, kernel: Callable):
        """
        
        Parameters
        ----------
        C : float
            Soft margin coefficient.
        kernel : Callable
            Функция ядра.
        
        """
        self.C = C
        self.kernel = kernel
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp
        
        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X 
            (можно считать, что равны -1 или 1). 
        
        """
        eps = 1e-5

        n_samples = X.shape[0]
        
        K = np.zeros((n_samples, n_samples))
        self.X = X
        self.y = y
        y = y.astype(float).reshape(-1, 1)
        for i, x in enumerate(X):
            K[i, :] = self.kernel(X, x)
 
        P = matrix((y.T * y) * K)
        Ones = np.ones(n_samples)
        Id_ = np.eye(n_samples)
        Zero = np.zeros(n_samples)
        q = matrix(-Ones)
        A = matrix(y.T)
        b = matrix(0.0)

        if self.C is None:
            G = matrix(-Id_)
            h = matrix(Zero)
        else:

            G = matrix(np.vstack((-Id_,
                                  Id_)))
            h = matrix(np.hstack((Zero,
                                  Ones * self.C)))

        solution = solvers.qp(P, q, G, h, A, b)
        self.alpha = np.array(solution['x'])
        self.support = np.arange(n_samples)[self.alpha[:, 0] > eps]
        self.b = (y[self.support] - (
                 (y * self.alpha) * K[:, self.support]).sum(
                  axis=0)).mean() 

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X 
            (т.е. то число, от которого берем знак с целью узнать класс).     
        
        """

        #Kernel = np.zeros((self.X.shape[0], X.shape[0]))
        res = np.zeros(X.shape[0]) + self.b
        for i in range(self.X.shape[0]):
            #Kernel[i, :] = self.kernel(X, self.X[i])
            res += (self.alpha.T * self.y)[0][i]*self.kernel(X, self.X[i])
        #res = (self.y * self.alpha * Kernel).sum(axis=0) + (
        #       self.b)

        return res
        
        
        res = (self.y * self.alpha * self.kernel(self.X, X)).sum(axis=0) + (
               self.b.mean())
        return res
        


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.   
        
        """
        return np.sign(self.decision_function(X))
