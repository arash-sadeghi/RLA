import numpy as np
class one1():
    def __init__(self):
        self.one1=1
        self.two()
    def two(self):
        self.two2=2

class one2(one1):
    def __init__(self):
        self.one2=2
        super().two()
class one3():
    def __init__(self):
        self.one3=3
        self.x=self.one3



O1=one1()
O2=one2()

O3=one3()
print(O3.one3,O3.x)
O3.one3=5
print(O3.one3,O3.x)

a=np.array([1,2])
b=np.copy(a)
a[0]*=2

print(a,b)














