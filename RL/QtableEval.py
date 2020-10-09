import numpy as np
with open('Qtable Thu Oct  8 21_24_39 2020\\0 Qtable Thu Oct  8 21_24_39 2020 .npy','rb') as Qtable:
    a=np.load(Qtable)
    x=[]
    for v in a:
        x.append(any(v>0))
print(sum(sum(a)))
# print(x)
# print(x.count(True))
