import numpy as np
import signal
# print('hi')
# class x:
#     def __init__(self):
#         self.xv='xv'
#     def tell(self):
#         print('I am father')
#     def xvset(self,v):
#         self.xv=v

# class y(x):
#     def __init__(self):
#         x.__init__(self)
#         # pass
#         # super().__init__()
#         # self.yv='yv'
#         # self.a=x.xv
#     def printer(self):
#         print(self.xv)
#         # print(x.xv)

# yy=y()
# xx=x()
# yy.printer()
# print('hi')
###################
# class x:
#     def __init__(self):
#         self.xv='xv'

#     class y:
#         def printer(self):
#             print(self.xv)
#         def changer(self):
#             self.xv='1'
# xx=x()
# print('hi')
###################
# def keyboardInterruptHandler(signal=None, frame=None):
#     print('[+] half data saved')
    
# signal.signal(signal.SIGINT, keyboardInterruptHandler)  

# while True:
#     keyboardInterruptHandler()
#     pass
###################
while True:
    x=input('ver')
    if x=='x':
        raise NameError('HiThere')