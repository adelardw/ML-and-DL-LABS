import numpy as np
import copy
from typing import List, NoReturn
import torch
from torch import nn
import torch.nn.functional as F


# Task 1

class Module:
    """
    Абстрактный класс. Его менять не нужно. Он описывает общий интерфейс
    взаимодествия со слоями нейронной сети.
    """
    def forward(self, x):
        pass

    def backward(self, d):
        pass

    def update(self, alpha):
        pass


class Linear(Module):
    """
    Линейный полносвязный слой.
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            Размер входа.
        out_features : int 
            Размер выхода.

        Notes
        -----
        W и b инициализируются случайно.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.uniform(0, 0.001, size=(self.in_features,
                                                   self.out_features))
        self.b = self.b = np.random.uniform(0, 0.001, size=(self.out_features))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Wx + b.

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
            То есть, либо x вектор с in_features элементов,
            либо матрица размерности (batch_size, in_features).
        Return
        ------
        y : np.ndarray
            Выход после слоя.
            Либо вектор с out_features элементами,
            либо матрица размерности (batch_size, out_features)

        """
        self.x = np.array(x)
        return self.x @ self.W + self.b

    def backward(self, d: np.ndarray) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """

        self.d = d
        self.loss_grad_x = self.d @ self.W.T
        self.loss_grad_w = self.x.T @ self.d
        self.loss_grad_b = self.d.sum(axis=0)
        return self.loss_grad_x

    def update(self, alpha: float) -> NoReturn:
        """
        Обновляет W и b с заданной скоростью обучения.

        Parameters
        ----------
        alpha : float
            Скорость обучения.
        """

        self.W -= alpha * self.loss_grad_w
        self.b -= alpha * self.loss_grad_b


class ReLU(Module):
    """
    Слой, соответствующий функции активации ReLU. Данная функция возвращает 
    новый массив, в котором значения меньшие 0 заменены на 0.
    """
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = max(0, x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        self.x = np.array(x)
        return np.maximum(0, self.x)

    def backward(self, d) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        self.d = d

        self.loss_grad_x = self.d * np.maximum(0, np.sign(self.x))
        return self.loss_grad_x


class CrossEntropyLoss(Module):
    def __init__(self):
        pass

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        GuideLine for future users:
        CE = (1[y = c]*ln(softmax(x_c))).sum()/BS
        1[y = c] -> OHE: create ohe = np.zeros(x.shape)
        x.shape <- (batch_size, num_features)
        y.shape <- (batch_size,)
        num_features = num_classes (in this layer)
        calculate ohe: ohe[range(len(x.shape[0])), y] = 1
        for example:
        y = [0,1,1,0,2] then:
        ohe = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]] 
        ohe[range(len(x.shape[0])), y] = 1
        ohe -> [[1,0,0],[0,1,0],[0,1,0],[1,0,0],[0,0,1]]
        But, its useless.        
        The easiest way take a: ln(softmax(x_c))[range(len(x.shape[0])), y]
        ln(softmax(x_c))[range(len(x.shape[0])), y] = ohe * ln(softmax(x_c)

        '''
        self.y_pred = np.exp(x)/np.exp(x).sum(axis=1, keepdims=True)
        self.y = np.array(y, dtype='uint8')

        return (np.log(np.exp(x).sum(axis=1, keepdims=True)) - x)[range(len(y)),
                                                                  self.y].sum()/x.shape[0]

    def backward(self) -> np.ndarray:
        '''
        D(CE) = (softmax(x_c) - (1[y = c])
        Alternative:
        make a copy of softmax(x_c)
        take a t = softmax(x_c)
        t[range(len(x.shape[0])), y] <- t[range(len(x.shape[0])), y] - 1
        return t
        '''
        copy_prob = self.y_pred.copy()
        copy_prob[range(len(self.y)), self.y] -= 1
        return copy_prob/self.y.shape[0]


# Task 2

class MLPClassifier:
    def __init__(self, modules: List[Module], epochs: int = 40,
                 alpha: float = 0.01, batch_size: int = 32):
        """
        Parameters
        ----------
        modules : List[Module]
            Cписок, состоящий из ранее реализованных модулей и 
            описывающий слои нейронной сети. 
            В конец необходимо добавить Softmax.
        epochs : int
            Количество эпох обучения.
        alpha : float
            Cкорость обучения.
        batch_size : int
            Размер батча, используемый в процессе обучения.
        """
        self.modules = modules
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает нейронную сеть заданное число эпох. 
        В каждой эпохе необходимо использовать cross-entropy loss для обучения, 
        а так же производить обновления не по одному элементу,
        а используя батчи
        (иначе обучение будет нестабильным и полученные
        результаты будут плохими.

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения.
        y : np.ndarray
            Вектор меток классов для данных.
        """

        self.loss_history = []
        n_batches = int(np.ceil(X.shape[0] / self.batch_size))

        data = np.hstack((np.array(X), np.array(y).reshape(-1, 1)))
        loss = CrossEntropyLoss()
        for _ in range(self.epochs):
            np.random.shuffle(data)
            ndata = [data[self.batch_size * i: self.batch_size * (i + 1)]
                     for i in range(n_batches)]
            for batch in ndata:
                X, y = batch[:, :-1], batch[:, -1]
                loss_per_batch = []
                out = X
                for layers in self.modules:
                    out = layers.forward(out)

                loss_f = loss.forward(out, y)
                loss_per_batch.append(loss_f)
                grad = loss.backward()

                for layers in self.modules[::-1]:
                    grad = layers.backward(grad)
                    layers.update(self.alpha)
            self.loss_history.append(sum(loss_per_batch)/n_batches)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.
        Return
        ------
        np.ndarray
            Предсказанные вероятности классов для всех элементов X.
            Размерность (X.shape[0], n_classes)

        """
        out = X
        for layers in self.modules:
            out = layers.forward(out)

        return np.exp(out)/np.exp(out).sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """
        Предсказывает метки классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Вектор предсказанных классов

        """
        p = self.predict_proba(X)

        return np.argmax(p, axis=1)

    def getloss(self):
        return np.array(self.loss_history)

# Task 3


classifier_moons = MLPClassifier([Linear(2, 2)])
# Нужно указать гиперпараметры
classifier_blobs = MLPClassifier([Linear(2, 3)])
# Нужно указать гиперпараметры


# Task 4

class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96,
                               kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=128,
                               kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=144,
                               kernel_size=(3, 3))
        # self.bn1 = nn.BatchNorm2d(num_features = 64)
        # self.bn2 = nn.BatchNorm2d(num_features = 144)
        # self.dpout = nn.Dropout(p=0.5)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.flt = nn.Flatten()
        self.linear1 = nn.Linear(1296, 128)
        self.linear2 = nn.Linear(128, 30)
        self.linear3 = nn.Linear(30, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x) 
        x = self.conv2(x) 
        # x = self.bn1(x)
        x = self.maxp(x) 
        x = F.silu(x)
        # x = self.dpout(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxp(x)
        x = F.silu(x)

        # x = self.dpout(x)
        x = self.conv5(x)
        # x = self.bn2(x)

        x = self.flt(x)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        x = F.silu(x)
        x = self.linear3(x)
        # x = F.softmax(x,dim=1)
        return x

    def load_model(self):
        """
        Используйте torch.load, чтобы загрузить обученную модель
        Учтите, что файлы решения находятся не в корне директории, поэтому необходимо использовать следующий путь:
        `__file__[:-7] +"model.pth"`, где "model.pth" - имя файла сохраненной модели `
        """
        pass

    def save_model(self):
        """
        Используйте torch.save, чтобы сохранить обученную модель
        """
        pass


def calculate_loss(X: torch.Tensor, y: torch.Tensor, model: TorchModel):
    """
    Cчитает cross-entropy.

    Parameters
    ----------
    X : torch.Tensor
        Данные для обучения.
    y : torch.Tensor
        Метки классов.
    model : Model
        Модель, которую будем обучать.

    """
    y_new = model(X)
    return F.cross_entropy(y_new, y)