import numpy as np
import pandas
import random
import copy
import re

# Task 1

def cyclic_distance(points, dist):
    
    neigh_dist = [dist(points[idx], points[idx + 1]) for idx in range(len(points) - 1)]
    neigh_dist.append(dist(points[0], points[-1]))
    return np.array(neigh_dist).sum()
    

def l2_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def l1_distance(p1, p2):
    return np.absolute(p2 - p1).sum()


# Task 2

class HillClimb:
    def __init__(self, max_iterations, dist):
        self.max_iterations = max_iterations
        self.dist = dist # Do not change
    
    def optimize(self, X):
        return self.optimize_explain(X)[-1]
    
    def optimize_explain(self, X):
        self.X = X
        return self.find_permutation_rec(
            [], np.arange(len(X)), cyclic_distance(X, self.dist), 0)
    
    def find_permutation_rec(self, perms, cur_perm, cur_dist, iteration):
        if iteration > self.max_iterations:
            return perms

        best_perm = cur_perm
        best_dist = cur_dist
        for i in range(len(self.X)):
            for j in range(i + 1, len(self.X)):
                new_perm = cur_perm.copy()
                new_perm[i], new_perm[j] = cur_perm[j], cur_perm[i]
                    
                new_dist = cur_dist - self.four_dists(cur_perm, i, j)
                new_dist = new_dist + self.four_dists(new_perm, i, j)

                if new_dist < best_dist:
                    best_dist = new_dist
                    best_perm = new_perm

                    perms.append(new_perm)
                    return self.find_permutation_rec(perms, best_perm, best_dist, iteration + 1)
        
        return perms

    def four_dists(self, perm, i, j):
        d1, d2, d3, d4, = 0, 0, 0, 0

        d1 = self.dist(self.X[perm][i], self.X[perm][(i-1)% len(self.X)])
        d2 = self.dist(self.X[perm][i], self.X[perm][(i+1)% len(self.X)])
        d3 = self.dist(self.X[perm][j], self.X[perm][(j-1)% len(self.X)])
        d4 = self.dist(self.X[perm][j], self.X[perm][(j+1)% len(self.X)])
            
        return d1 + d2 + d3 + d4

        

# Task 3

class Genetic:
    def __init__(self, iterations, population, survivors, distance):
        self.pop_size = population
        self.surv_size = survivors
        self.dist = distance
        self.iters = iterations
    
    def optimize(self, X):
        pass
    
    def optimize_explain(self, X):
        pass

    
# Task 4

class BoW:
    def __init__(self, X: np.ndarray, voc_limit: int = 1000):
        """
        Составляет словарь, который будет использоваться для векторизации предложений.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ), 
            по которому будет составляться словарь.
        voc_limit : int
            Максимальное число слов в словаре.

        """
        self.X = X 
        self.voc_lim = voc_limit
        self.n_sentences = self.X.shape[0]
        sentences = []

        for x in X:
            sentences.extend(self.processing(x))

        #voc_limit = 100
        vocab, word_freq = np.unique(np.asarray(sentences), return_counts=True)
        freqs = (word_freq).argsort()[::-1]
        vocab = vocab[freqs][:self.voc_lim]
        self.codes = {word: k for k, word in enumerate(vocab)}
        
    def processing(self, sentence):
        res = []
        u = re.split(r'[\d_\\.())&=?!\^d]', sentence, maxsplit = 7)
        for h in u:
            k = re.sub(r'[.,!?\'\"()]', '', h).lower().split() 
            if len(k) > 3:
                res.extend(k)
        return res
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Векторизует предложения.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ), 
            который необходимо векторизовать.
        
        Return
        ------
        np.ndarray
            Матрица векторизованных предложений размерности (n_sentences, vocab_size)
        """
        n_sentences = X.shape[0]
        X_vector = np.zeros((n_sentences, self.voc_lim), dtype=int)
        for i, sentence in enumerate(X):
            sentence = self.processing(sentence)
            row = np.zeros(self.voc_lim, dtype=int)
            for word in sentence:
                if word in self.codes.keys():
                    idx = self.codes[word]
                    row[idx] += 1
                    X_vector[i, :] = row.copy()
        
        return X_vector

# Task 5

class NaiveBayes:
    def __init__(self, alpha: float):
        self.alpha = alpha
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes, counts = np.unique(y, return_counts=True)
        self.prob_y = np.log(counts/X.shape[0])
        feat = X.shape[1]
        p_hat = np.zeros((self.classes.shape[0], feat))
        
        for i, label in enumerate(self.classes):
            X_class = X[y==label]
            div = np.sum(X_class)
            for j in range(feat):
                p_hat[i, j] = (np.sum(X_class[:,j]) + self.alpha) / (div + self.alpha * feat)
        
        self.log_p_hat = np.log(p_hat)
        
    def predict(self, X: np.ndarray):
        return [self.classes[i] for i in np.argmax(self.log_proba(X), axis=1)]
    
    def log_proba(self, X: np.ndarray) -> np.ndarray:
        n_classes = self.classes.shape[0]
        matrix = np.zeros((X.shape[0], n_classes))
        
        for i, elem in enumerate(X):
            for j in range(n_classes):
                matrix[i, j] = np.sum(self.log_p_hat[j] * elem) + self.prob_y[j]
          
        return matrix