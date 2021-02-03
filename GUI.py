import tkinter as tk
from math import sqrt
class parameterGUI(object):
    def __init__(self,flags,vals):
        self.root= tk.Tk()
        self.root.title("Parameter determination")
        self.frame = tk.LabelFrame(self.root, padx=50, pady=50)
        self.frame.pack(padx=10, pady=10)

        self.canvas = tk.Canvas(self.frame, width = 600, height = 500)
        self.canvas.pack()

        self.OK = tk.Button(text='OK',bg='white', command=self.OK,font= 10,width=8)
        self.canvas.create_window(600, 400, window=self.OK)

        self.flags=flags
        
        self.fvar=[tk.IntVar() for _ in range(len(self.flags))]
        self.labels=[0 for _ in range(len(self.flags))]
        self.cb=[0 for _ in range(len(self.flags))]

        for c,v in enumerate(self.flags):
            self.labels[c]=tk.Label(text=v[0],font=("Times","12","bold"))
            self.canvas.create_window(400, 10+c*30, window=self.labels[c],anchor=tk.W)

            self.fvar[c].set(v[1])
            self.cb[c]=tk.Checkbutton(variable=self.fvar[c],textvariable=self.fvar[c])
            self.canvas.create_window(600, 10+c*30, window=self.cb[c],anchor=tk.W)

        self.vals=vals

        self.labels=[0 for _ in range(len(self.vals))]
        self.e=[0 for _ in range(len(self.vals))]

        for c,v in enumerate(self.vals):
            self.labels[c]=tk.Label(text=v[0],font=("Times","12","bold"))
            self.canvas.create_window(10, 10+c*30, window=self.labels[c],anchor=tk.W)

            self.e[c]=tk.Entry()
            self.e[c].insert(0,str(v[1]))
            self.canvas.create_window(200, 10+c*30, window=self.e[c],anchor=tk.W)
        self.root.mainloop()
        # self.label0=tk.Label(text="Welcome to My Screen Recorder",font=("Times", "24", "bold italic"))
        # self.canvas.create_window(200, 50, window=self.label0)


        # self.loc = tk.Entry(width=loc_w)
        # self.loc.insert(0,self.save_path)
        # self.canvas.create_window(200, 200, window=self.loc)

        # self.browse_b =tk.Button(text='browse',bg='white', command=self.browse,font= 10,width=8)
        # self.canvas.create_window(400, 200, window=self.browse_b)


    def OK(self):

        flag_vals=list(map(lambda x: x.get(),self.fvar))
        for c,v in enumerate(self.flags):
            self.flags[c][1]=flag_vals[c]

        v_vals=list(map(lambda x: x.get(),self.e))
        for c,v in enumerate(self.vals):
            self.vals[c][1]=v_vals[c]
        self.root.destroy()
