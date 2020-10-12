# import numpy as np
# class one1():
#     def __init__(self):
#         self.one1=1
#         self.two()
#     def two(self):
#         self.two2=2

# class one2(one1):
#     def __init__(self):
#         self.one2=2
#         super().two()
# class one3():
#     def __init__(self):
#         self.one3=3
#         self.x=self.one3



# O1=one1()
# O2=one2()

# O3=one3()
# print(O3.one3,O3.x)
# O3.one3=5
# print(O3.one3,O3.x)

# a=np.array([1,2])
# b=np.copy(a)
# a[0]*=2

# print(a,b)
#..................................................
# a=[]
# a.append(3)
# print(a)
#..................................................
import numpy
import numpy as np
import matplotlib.pyplot as plt

x = numpy.array([1, 7, 20, 50, 79])
y = numpy.array([10, 19, 30, 35, 51])
Z=numpy.polyfit(numpy.log(x), y, 1)
z=Z[0]*np.log(x)+Z[1]
# plt.plot(x,label='x')
plt.plot(x,y,label='y')
plt.plot(x,z,label='z')
plt.legend()
plt.xlim(0,80)
plt.ylim(10,50)
plt.show()
# array([ 8.46295607,  6.61867463])










