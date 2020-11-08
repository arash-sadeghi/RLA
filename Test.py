# class x:
#     def __init__(self):
#         self.xv='xv'

#     def printer(self):
#         print(self.xv)
#     def changer(self):
#         self.xv='1'
# xx=x()
# # Step 1
import pickle
 
 
# # Step 2
# with open('xx.hi', 'wb') as xx_file:
#    # Step 3
#   pickle.dump(xx, xx_file)

with open('x.hi', 'rb') as xx_file:
 
    # Step 3
    XX = pickle.load(xx_file)
 
    # After config_dictionary is read from file
    print(XX)



