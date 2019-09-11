#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import numpy as np
from scipy.optimize import linprog


# In[2]:


def get_cell_type(row, col):
    #function to determine the type of cell
    if ((row == 0 or row == 4) and (col == 0 or col == 4)):
        return 0, 0.5; #corner
    elif(row == 0 or col == 0 or row == 4 or col == 4):
        return 1, 0.25; #edge
    else:
        return 2, 0; #middle
    
def get_neighbors(row, col):
    #function to determine the valid neighbours of a given state/cell
    arr = []
    if (row+1<5):
        arr.append((row+1, col))
    if (col+1<5):
        arr.append((row, col+1))
    if (col-1>=0):
        arr.append((row, col-1))
    if (row-1>=0):
        arr.append((row-1, col))
    return arr

def get_state(row, col, action): 
    #returns the next state the action would lead to from a given state
    #returns reward obtained as the 3rd argument
    if (row == 0 and col == 1):
        return (4, 1, 10)
    elif (row == 0 and col == 3):
        return (2, 3, 5)
    if (action == 0): #up
        if (row - 1 >= 0):
            return (row - 1, col, 0)
        
    if (action == 1): #right
        if (col + 1 < 5):
            return (row, col+1, 0)
        
    if (action == 2): #down
        if (row + 1 < 5):
            return (row + 1, col, 0)

    if (action == 3): #left
        if (col - 1 >= 0):
            return (row, col-1, 0)
    
    return (row, col, -1)
        


# In[3]:


A = [[0 for i in range(25)] for j in range(25)]
b = [0 for i in range(25)]


# In[4]:


a = get_neighbors(4,0)
print(a)


# In[5]:


row_num = 0;
for row in range(5):
    for col in range(5):
        cell_type, bval = get_cell_type(row, col)
        arr_ind = 5*row + col
        b[arr_ind] = bval
        neighbors = get_neighbors(row, col)
        if (row == 0 and col == 1):
            b[arr_ind] = -10
            A[row_num][arr_ind] = -1
            A[row_num][21] = 0.9
        elif (row == 0 and col == 3):
            b[arr_ind] = -5
            A[row_num][arr_ind] = -1
            A[row_num][13] = 0.9
        elif (cell_type == 0):
            A[row_num][arr_ind] = -0.55
            for neighbor in neighbors:
                index = 5*neighbor[0] + neighbor[1]
                A[row_num][index] = 0.25*0.9
        elif (cell_type == 1):
            A[row_num][arr_ind] = 0.25*0.9 - 1
            for neighbor in neighbors:
                index = 5*neighbor[0] + neighbor[1]
                A[row_num][index] = 0.25*0.9
        else:
            A[row_num][arr_ind] = -1
            for neighbor in neighbors:
                index = 5*neighbor[0] + neighbor[1]
                A[row_num][index] = 0.25*0.9
        row_num += 1


# In[6]:


Ainv = numpy.linalg.inv(A)


# In[7]:


b = numpy.array(b)


# In[8]:


ans = numpy.matmul(Ainv, b)
print(numpy.reshape(numpy.round(ans, 1), (5, 5)))

