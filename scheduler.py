import os
dirs=os.listdir()
for dir in dirs:
    if "%" in dir:
        print("EXECUTING",dir)
        os.system("python3 "+dir+"/MAIN.py")