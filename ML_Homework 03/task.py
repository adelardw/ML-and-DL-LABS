
import numpy as np
from sklearn.preprocessing import StandardScaler


# Task 1

def mse(y_true:np.ndarray, y_predicted:np.ndarray):

    return ((y_true-y_predicted)**2).mean()

def r2(y_true:np.ndarray, y_predicted:np.ndarray):
    return 1- mse(y_true,y_predicted)/mse(y_true,y_true.mean())

# Task 2

class NormalLR:
    def __init__(self):
        self.weights = None # Save weights here
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        bias = np.ones(X.shape[0]).reshape(-1,1)
        X = np.hstack((bias,X))
        self.weights = (np.linalg.inv((X.T @ X)) ) @ (X.T @ y)
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        y_pred =  X @ self.weights[1:] + self.weights[0]
        return y_pred
    
# Task 3

class GradientLR:
    def __init__(self, alpha:float, iterations=10000, l=0.):
        self.weights = None # Save weights here
        self.alpha = alpha
        self.iterations = iterations
        self.l = l
    
    def grad(self, X: np.ndarray, y:np.ndarray,w:np.array)-> np.ndarray:

        return (2*(X.T @ (X @ w - y))/len(y) + self.l*np.sign(w))

    def fit(self, X:np.ndarray, y:np.ndarray):
        bias = np.ones(X.shape[0]).reshape(-1,1)
        X = np.hstack((bias,X))
        #np.random.seed(13)
        self.weights = np.random.rand(X.shape[1])
        #self.weights = np.zeros(X.shape[1])

        for i in range(self.iterations):
            self.weights -=  self.alpha* self.grad(X, y,w = self.weights )
    
        
    def predict(self, X:np.ndarray):
        y_pred =  X @ self.weights[1:] + self.weights[0]
        return y_pred

# Task 4

def get_feature_importance(linear_regression):
    w = np.abs(linear_regression.weights[1:])

    return w/ w.sum()

def get_most_important_features(linear_regression):

    return np.argsort(np.abs(linear_regression.weights)[1:])[::-1]