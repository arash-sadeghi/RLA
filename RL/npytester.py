import os
import numpy as np
print(os.listdir())
with open('Mon Oct 12 19_12_45 2020 RL/results Mon Oct 12 19_17_58 2020.npy','rb') as f:
    data=np.load(f)
    print(np.shape(data))
