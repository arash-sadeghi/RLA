'''
class x address <class '__main__.x'>
instance X addres <__main__.x object at 0x7f035eb576a0>
passed class x address in y: self.x :  <__main__.x object at 0x7f035eb576a0>
father x address in y: x : <class '__main__.x'>

conclusion: self.x inside y has the address of created object, X. any cahnges 
in X will apply to self.x they have same addres
'''
from termcolor import colored as c

# print(c('hi {}','red').format(5))

class x:
    def __init__(self):
        self.f=5
        self.Y=y(self)
        self.s=6

    def hiiiiii(self):
        return 'hi'
    def change(self):
        self.s=8
class y(x):
    def __init__(self,x):
        self.t=2
        self.x=x
    def printer(self):
        # print(self.x.f,self.x.s,'\n',c(dir(self.x),'blue'))
        print('passed class x address in y: self.x : ',self.x)
        print('father x address in y: x :',x)

# X=x()
# X.Y.printer()


X=x()
print('class x address',x)
print('instance X addres',X)
X.Y.printer()

# print(X.hiiiiii())
# print(c(X.__dict__,'green'))
# print(dir(X))