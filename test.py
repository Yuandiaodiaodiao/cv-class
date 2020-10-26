import numpy as np

a=np.array([1,2,2,3,4,4,4,4])
f=np.vectorize(lambda x:x+1)
print(f(a))
