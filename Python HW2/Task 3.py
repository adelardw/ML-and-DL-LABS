#! bin/python
import numpy as np


A = np.random.randint(1,10000,(128,2))
def euclid(A: np.array) -> np.array:
  n = A.shape[0]
  res = np.zeros(n,dtype = np.int32)
  for i,[a, b] in enumerate(A):
    if a == 0:
      res[i] = b
    if b == 0:
      res[i] == a

    else:
      while a != b:
        if a > b:
          a -= b
        else:
          b -= a
      res[i] = a

  return res

if __name__ == "__main__":

    print( euclid(A) )


