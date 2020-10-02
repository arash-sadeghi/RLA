# class hi():
#     def __init__(self):
#         self.strng='hi'
#     def talk(self):
#         print(self.strng)
#     def what(self,s):
#         self.strng=s
# o=hi()
# o.what('fa')
# o.talk()
# del o
# o=hi()
# o.talk()
import numpy as np

with open("results.npy",'rb') as f:
    a=np.load(f)
    print(a)