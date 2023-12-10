#! bin/python
import numpy as np
from typing import NoReturn,Union



class MyException(Exception):
    
    def __init__(self,message = 'Matrix:shape[0] must be < Matrix:shape[1]'):
      self.message = message
      print(message)


class Solution:


  def __init__(self,A: np.array)-> NoReturn:

    self.A = A
    self.m = self.A.shape[1]
    self.n = self.A.shape[0]
    if self.m > self.n:
      self.frddgrees = self.m - self.n
    elif self.m - 1 == self.n:
      self.frddgrees = 0
    else:
      raise MyException

    self.rank = self.m - 1

  def gauss_solution(self) -> np.array:
    self.A = self.A.astype(np.float64)
    n = self.A.shape[0]
    m = self.A.shape[1]

    j = 0
    for i,x in enumerate(self.A):

        if j < self.m:
          if x[j] != 0:
            x /= x[j]
          elif i + 1 != n:
            self.A[[i, i + 1]] = self.A[[i + 1, i]]
            self.rank -= 1
            x = self.A[i]
          for r in self.A[(i + 1):]:
                r -= x*r[j]
          j += 1
        else:
            break

    self.A = np.flip(self.A,axis = 0)
    j = self.m - self.n + 1

    for i,x in enumerate(self.A):
        if j < self.m:
            for r in self.A[(i + 1):]:
                r -= x*r[self.m - j]
            j += 1
        else:
            break

    self.A = np.flip(self.A,axis = 0)
    return self.A

  def is_single(self)-> bool:
      if self.frddgrees == 0:
        return True

  def freedom_degrees(self) -> int:
    if self.is_single():
      return 0
    else:
      return self.frddgrees

  @classmethod
  def solution(cls, res: np.array)-> np.array:
    fred = cls(res).freedom_degrees()
    if cls(res).is_single():
      return cls(res).gauss_solution()[:,-1:]
    else:
      return cls(res).gauss_solution()[:,-fred:]


def result(matrix: np.array)->Union[np.array,None]:
  try:
      return Solution(matrix).solution(matrix)
  except MyException:
      return None
  
np.random.seed(13)
G = np.random.randint(-10,4,(4,3))

if __name__ =='__main__':
   print(result(G))