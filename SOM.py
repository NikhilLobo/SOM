"""
Created on Sun Feb 15 22:53:06 2020

@author: Nikhil Lobo
@email: nikhillobo007@gmail.com
"""




from tkinter import * 
from tkinter.ttk import *
from tkinter import messagebox
import numpy as np
import graphics as g
import pandas as pd



"""Creation of GUI using Tkinter."""
window = Tk()

window.title("Self Organizing Maps")

window.geometry('400x750')
lbl1 = Label(window, text="Enter the dataset file name :")
lbl1.place(x=30, y=30)
txt1 = Entry(window ,width=15)
txt1.place(x=230, y=30)
txt1.focus()

lbl2 = Label(window, text="Enter the dimension of Map :")
lbl2.place(x=30, y=80)
txt2 = Entry(window,width=15)
txt2.place(x=230, y=80)

lbl3 = Label(window, text="Enter the number of nodes :")
lbl3.place(x=30, y=130)
txt3 = Entry(window,width=15)
txt3.place(x=230, y=130)

lbl4 = Label(window, text="Enter the Learning Rate :")
lbl4.place(x=30, y=180)
txt4 = Entry(window,width=15)
txt4.place(x=230, y=180)

lbl5 = Label(window, text="Enter the number of Epochs :")
lbl5.place(x=30, y=230)
txt5 = Entry(window,width=15)
txt5.place(x=230, y=230)


lbl6 = Label(window, text="Do want animation.? :")
lbl6.place(x=30, y=280)
combo = Combobox(window,width=14)
combo['values']= ("Yes", "No")
combo.place(x=230, y=280)


lbl7 = Label(window, text="Enter the 1st feature name :")
lbl7.place(x=30, y=440)
    
    
txt7 = Combobox(window,width=14)
txt7.place(x=230, y=440)


lbl8 = Label(window, text="Enter the 2nd feature name :")
lbl8.place(x=30, y=490)
    
txt8 = Combobox(window,width=14)
txt8.place(x=230, y=490)


lbl9 = Label(window, text="Dependent Value :")
lbl9.place(x=30, y=390)

txt9 = Combobox(window,width=14)
txt9.place(x=230, y=390)


"""Defining the initial values. """
file=""                      
dimension=0
nodes=0
learning_rate=0
epochs=0
animation=""


def find_bmu1(t, net, m,d):
    """
        Find the best matching unit for a given vector for 1- dimensional map.
    """
    bmu_idx = np.array([0])
    min_dist = np.iinfo(np.int).max
    
    # calculate the distance between each neuron and the input vector
    for x in range(net.shape[0]):
            w = net[x,:].reshape(m, 1)
            
            if d!="None":
                w[d]=0
                t[d]=0 
            
            sq_dist = np.sum((w - t) ** 2)
            sq_dist = np.sqrt(sq_dist)
            if sq_dist < min_dist:
                min_dist = sq_dist # dist
                bmu_idx = np.array([x]) # id
    
    bmu = net[bmu_idx[0], :].reshape(m, 1)
    return (bmu, bmu_idx)

def find_bmu2(t, net, m,d):
    """
        Find the best matching unit for a given vector for 2- dimensional map.
    """
    bmu_idx = np.array([0, 0])
    min_dist = np.iinfo(np.int).max
    
    # calculate the distance between each neuron and the input vector
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w = net[x, y, :].reshape(m, 1)
            if d!="None":
                w[d]=0
                t[d]=0
           
            sq_dist = np.sum((w - t) ** 2)
            sq_dist = np.sqrt(sq_dist)
            if sq_dist < min_dist:
                min_dist = sq_dist # minimum distance calculation 
                bmu_idx = np.array([x, y]) #BMU  id
    
    bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(m, 1)
    return (bmu, bmu_idx)



"""Finding the neighborhood of BMU and updating their weights."""

def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)

def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * np.exp(-i / n_iterations)

def calculate_influence(distance, radius):
    return np.exp(-distance / (2* (radius**2)))


"""Reading the user inputs and setting the values. """
def clicked():
    
    file=txt1.get()+".csv"
    dimension=int(txt2.get())
    nodes=txt3.get()
    learning_rate=float(txt4.get())
    init_learning_rate=learning_rate
    epochs=txt5.get()
    n_iterations = int(epochs)
    animation=combo.get()
    time_constant = n_iterations

    df=pd.read_csv("data/"+file,encoding='utf8',engine='python')
    arr=[]
    arr1=["None"]
    for col in df.columns: 
        arr.append(col)
        arr1.append(col)

    txt7['values'] = (arr)
    txt8['values'] = (arr)
    txt9['values'] = (arr1)
    
    messagebox.showinfo('Please wait the map is getting train')


btn1 = Button(window, text="train Map",width = 20,command=clicked)
btn1.place(x=100,y=330)


"""Reading features and other parameters for displaying the map in different window on each click on the visualization button """

def visualize():

    
    file=txt1.get()+".csv"
    dimension=int(txt2.get())
    nodes=txt3.get()
    learning_rate=float(txt4.get())
    init_learning_rate=learning_rate
    epochs=txt5.get()
    n_iterations = int(epochs)
    animation=combo.get()
    time_constant = n_iterations

    df=pd.read_csv("data/"+file,encoding='utf8',engine='python')

    arr=[]
    for col in df.columns:  
        arr.append(col)

    
    df1=df.to_numpy()
    dataset=df1.transpose() 
    data=dataset

    m = dataset.shape[0]
    n = dataset.shape[1]
    
    if dimension==1:
        network_dimensions = np.array([int(nodes)])
        net = np.random.random((network_dimensions[0], m))*100
        init_radius = network_dimensions[0] / 2
    
    if dimension==2:
        network_dimensions = np.array([int(nodes),int(nodes)])
        net = np.random.random((network_dimensions[0], network_dimensions[1], m))*100
        init_radius = network_dimensions[0] / 2

    
    #Feture reading
    f1=txt7.get() 
    f2=txt8.get()
    win = g.GraphWin(f1+'--'+f2+' Projection', 900, 900) # give title and dimensions
    win.yUp()
    win.setBackground("white")
  
    
    
    p=arr.index(f1)
    q=arr.index(f2)
 
    
    #Displaying the data
    for a in range(df1.shape[0]):
            val=df1[a]
            head = g.Point(val[p],val[q])
            head.setFill("black")
            head.draw(win)
    
    #Reading the dependent value for contarined topology map
    dd=txt9.get() 
    if dd!="None":
        dep=df.columns.get_loc(dd)
    else:
        dep=dd
    
    for i in range(n_iterations):
       
        # select a training example at random oine at a time.
        t = data[:, np.random.randint(0, n)].reshape(np.array([m, 1]))
        # find its Best Matching Unit
        
        if dimension ==1:
            bmu, bmu_idx = find_bmu1(t, net, m,dep)
        else:
            bmu, bmu_idx = find_bmu2(t, net, m,dep) 
        
        
        # decay the SOM parameters
        r = decay_radius(init_radius, i, time_constant)
        l = decay_learning_rate(init_learning_rate, i, n_iterations)
        
        
        # update weight vector to move closer to input
        # and move its neighbours in the space closer
        if dimension==1:    
            for x in range(net.shape[0]):
                    w = net[x,:].reshape(m, 1)
                    if dep!="None":
                        w[dep]=0
                        t[dep]=0
                    w_dist = np.sum((np.array([x]) - bmu_idx) ** 2)
                    w_dist = np.sqrt(w_dist)
                    
                    if w_dist <= r:
                        # calculate the degree of influence
                        influence = calculate_influence(w_dist, r)
                        # new w = old w + (learning rate * influence * delta)
                        # where delta = input vector (t) - old w
                        new_w = w + (l * influence * (t - w))
                        net[x,:] = new_w.reshape(1, m)
                        dim=net[x,:]
                        x1=dim[p]
                        y1=dim[q]
                        # Setting the parameters for animation
                        # coloring the nuerons to green when they in the neighborhood region.
                        head1 = g.Circle(g.Point(x1,y1),4)
                        head1.setFill("green")
                        head1.setOutline("white")
                    else:
                        dim=net[x,:]
                        x1=dim[p]
                        y1=dim[q]
                        # Setting the parameters for animation
                        # coloring the nuerons to red when they are outside the neighborhood region.
                        head1 = g.Circle(g.Point(x1,y1),3)
                        head1.setFill("red")
                        head1.setOutline("white") 
                        
                    if animation == "Yes":
                        head1.draw(win)
            
            # Connecting the all the neighbor nuerons using line
            if animation == "Yes":  
                
                for x in range(net.shape[0]):
                    if x<net.shape[0]-1:
                        dim1=net[x]
                        dim2=net[x+1]
                        head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(dim2[p],dim2[q]))
                        head.setFill("lavender")
                        head.setOutline("lavender")
                        head.draw(win)        
                for x in range(net.shape[0]):
                    if x<net.shape[0]-1:
                        dim1=net[x]
                        dim2=net[x+1]
                        head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(dim2[p],dim2[q]))
                        head.setFill("white")
                        head.setOutline("white")
                        head.draw(win)
                for a in range(net.shape[0]):
                    dim=net[a,:]
                    x=dim[p]
                    y=dim[q]
                    head1 = g.Circle(g.Point(x,y),3)
                    head1.setFill("white")
                    head1.setOutline("white")
                    head1.draw(win)
        
        else:
            for x in range(net.shape[0]):
                for y in range(net.shape[1]):
                    w = net[x,y,:].reshape(m,1)
                    if dep!="None":
                        w[dep]=0
                        t[dep]=0
                    w_dist = np.sum((np.array([x,y]) - bmu_idx) ** 2)
                    w_dist = np.sqrt(w_dist)
                    
                    
                    
                    if w_dist <= r:
                        # calculate the degree of influence (based on the 2-D distance)
                        influence = calculate_influence(w_dist, r)
                        # new w = old w + (learning rate * influence * delta)
                        # where delta = input vector (t) - old w
                        new_w = w + (l * influence * (t - w))
                        net[x, y,:] = new_w.reshape(1,m)
                        dim=net[x, y,:]
                        x1=dim[p]
                        y1=dim[q]
                      
                        head1 = g.Circle(g.Point(x1,y1),4)
                        head1.setFill("green")
                        head1.setOutline("white")
                    else:
                        dim=net[x,y,:]
                        x1=dim[p]
                        y1=dim[q]
                
                        head1 = g.Circle(g.Point(x1,y1),3)
                        head1.setFill("red")
                        head1.setOutline("white") 
                        
                    if animation == "Yes":
                        head1.draw(win)
             
            if animation == "Yes":  
                
                #connecting the neighbor
                for row in range(net.shape[0]):
                    for col in range(net.shape[1]):
                        if row == 0 and col < net.shape[1]-1:
                            dim1=net[row,col]
                            dim2=net[row,col+1]
                            head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(dim2[p],dim2[q]))   
                            head.setFill("lavender")
                            head.setOutline("lavender")
                            head.draw(win)  
                        if col ==net.shape[1]-1 and row <net.shape[0]-1:
                            dim1=net[row,col]
                            bottom=net[row+1,col]
                            head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(bottom[p],bottom[q]))     
                            head.setFill("lavender")
                            head.setOutline("lavender")
                            head.draw(win) 
                        
                        if row!=0 and col <net.shape[1]-1:
                            dim1=net[row,col]
                            front=net[row,col+1]
                            top=net[row-1,col]
                            head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(front[p],front[q]))
                            head.setFill("lavender")
                            head.setOutline("lavender")
                            head.draw(win) 
                            head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(top[p],top[q]))
                            head.setFill("lavender")
                            head.setOutline("lavender")
                            head.draw(win)
                        if row==net.shape[0]-1 and col==net.shape[0]-1:
                            dim1=net[row,col]
                            top=net[row-1,col]
                            head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(top[p],top[q])) 
                
                #clearing the map for the next plot.
                for row in range(net.shape[0]):
                    for col in range(net.shape[1]):
                        if row == 0 and col < net.shape[1]-1:
                            dim1=net[row,col]
                            dim2=net[row,col+1]
                            head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(dim2[p],dim2[q]))   
                            head.setFill("white")
                            head.setOutline("white")
                            head.draw(win)  
                        if col ==net.shape[1]-1 and row <net.shape[0]-1:
                            dim1=net[row,col]
                            bottom=net[row+1,col]
                            head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(bottom[p],bottom[q]))     
                            head.setFill("white")
                            head.setOutline("white")
                            head.draw(win) 
                        
                        if row!=0 and col <net.shape[1]-1:
                            dim1=net[row,col]
                            front=net[row,col+1]
                            top=net[row-1,col]
                            head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(front[p],front[q]))
                            head.setFill("white")
                            head.setOutline("white")
                            head.draw(win) 
                            head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(top[p],top[q]))
                            head.setFill("white")
                            head.setOutline("white")
                            head.draw(win)
                        if row==net.shape[0]-1 and col==net.shape[0]-1:
                            dim1=net[row,col]
                            top=net[row-1,col]
                            head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(top[p],top[q])) 
                            head.setFill("white")
                            head.setOutline("white")
                            head.draw(win)
                
                for a in range(net.shape[0]):
                    for b in range(net.shape[1]):
                        dim=net[a, b,:]
                        x=dim[p]
                        y=dim[q]
                        head1 = g.Circle(g.Point(x,y),3)
                        head1.setFill("white")
                        head1.setOutline("white")
                        head1.draw(win)
                        
    
            
    """ Final plot displaying based on the dimension"""        
    if dimension ==1:
        for a in range(net.shape[0]):
            dim=net[a,:]
            x=dim[p]
            y=dim[q]
            head1 = g.Circle(g.Point(x,y),3)
            head1.setFill("green")
            head1.setOutline("green")
            head1.draw(win)
                
        for x in range(net.shape[0]):
            if x<net.shape[0]-1:
                dim1=net[x]
                dim2=net[x+1]
                head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(dim2[p],dim2[q]))
                head.setFill("red")
                head.setOutline("red")
                head.draw(win)
    else:
        for a in range(net.shape[0]):
            for b in range(net.shape[1]):
                dim=net[a, b,:]
                x=dim[p]
                y=dim[q]
                head = g.Circle(g.Point(x,y),3)
                head.setFill("green")
                head.setOutline("green")
                head.draw(win) 
    
        for row in range(net.shape[0]):
                for col in range(net.shape[1]):
                    if row == 0 and col < net.shape[1]-1:
                        dim1=net[row,col]
                        dim2=net[row,col+1]
                        head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(dim2[p],dim2[q]))   
                        head.setFill("red")
                        head.setOutline("red")
                        head.draw(win) 
                        
                    if col ==net.shape[1]-1 and row <net.shape[0]-1:
                        dim1=net[row,col]
                        bottom=net[row+1,col]
                        head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(bottom[p],bottom[q]))     
                        head.setFill("red")
                        head.setOutline("red")
                        head.draw(win) 
                    
                    if row!=0 and col <net.shape[1]-1:
                        dim1=net[row,col]
                        front=net[row,col+1]
                        top=net[row-1,col]
                        head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(front[p],front[q]))
                        head.setFill("red")
                        head.setOutline("red")
                        head.draw(win) 
                        head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(top[p],top[q]))
                        head.setFill("red")
                        head.setOutline("red")
                        head.draw(win)
                        
                    if row==net.shape[0]-1 and col==net.shape[0]-1:
                        dim1=net[row,col]
                        top=net[row-1,col]
                        head=g.Line(g.Point(dim1[p],dim1[q]),g.Point(top[p],top[q])) 
                        head.setFill("red")
                        head.setOutline("red")
                        head.draw(win) 

                
    win.getMouse() # Pause to view result
    win.close()

btn2 = Button(window, text="Visualization",width = 20,command=visualize)
btn2.place(x=100, y=550)


window.mainloop()