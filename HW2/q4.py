#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy
import numpy as np
from scipy.optimize import linprog


# In[20]:


def get_cell_type(row, col):
    if ((row == 0 or row == 4) and (col == 0 or col == 4)):
        return 0, 0.5; #corner
    elif(row == 0 or col == 0 or row == 4 or col == 4):
        return 1, 0.25; #edge
    else:
        return 2, 0; #middle
    
def get_neighbors(row, col):
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

def get_state(row, col, action): #returning reward obtained as the 3rd argument
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
        


# In[56]:


def get_action_matrix(po):
    action_dict = {0:'U', 1:'R', 2:'D', 3:'L'}
    action_matrix = ["" for i in range(len(po))]
    optimal_actions = [["" for i in range (5)] for j in range(5)]
    for i in range(1, len(po)-1):
        values = po[i]
        maxval = max(values)
        for j, val in enumerate(values):
            if (val == maxval):
                action_matrix[i] = action_matrix[i] + action_dict[j]
    mat = numpy.reshape(action_matrix, (4, 4))
    return mat


# # Policy Iteration

# In[62]:


def get_next_state(state_num, action): #returning reward obtained as the 3rd argument
    row = state_num//4
    col = state_num%4
    if (row == 0 and col == 0):
        return (0, 0)
    elif (row == 3 and col == 3):
        return (3, 3)
    if (action == 0): #up
        if (row - 1 >= 0):
            return (row - 1, col)
        
    if (action == 1): #right
        if (col + 1 < 4):
            return (row, col+1)
        
    if (action == 2): #down
        if (row + 1 < 4):
            return (row + 1, col)

    if (action == 3): #left
        if (col - 1 >= 0):
            return (row, col-1)    
    return (row, col)


# In[67]:


v = [0]*16
num_states = 16
policy = [[0.25 for i in range(4)] for j in range(16)]
flag = True
terminal = [0, 15]
reward = -1
while (flag):
    flag2 = True
    while (flag2):
        delta = 0
        for state in range(num_states):
            val = v[state] 
            if (state in terminal):
                continue
            nval = 0
            for action in range(4):
                next_state = get_next_state(state, action)
                state_num = 4*next_state[0] + next_state[1]
                nval += policy[state][action]*(reward + v[state_num])
            v[state] = nval
            delta = max(delta, abs(val-v[state]))
        print("delta: ", delta)
        if delta < 0.00001:
            break
#     flag3 = True
  
    policy_stable = True
    for state in range(num_states):
        if (state in terminal):
            continue
        old_action = []
        for act in policy[state]:
            old_action.append(act)
#         old_action = policy[state]
        li = []
        for action in range(4):
            next_state = get_next_state(state, action)
            next_state = 4*next_state[0] + next_state[1]
            li.append(reward + v[next_state])
        li = numpy.round(numpy.array(li), 2)
        maxval = max(li)
        count=0;indices=[]


        for action in range(4):
            if (li[action] == maxval):
                indices.append(action)
                count+=1;
        for index in range(4):
            if index in indices:
                policy[state][index] = 1/count;
            else:
                policy[state][index] = 0
        for ind in range(len(old_action)):
            if (old_action[ind] != policy[state][ind]):
                policy_stable = False
    if (policy_stable):
        break
po = numpy.reshape(policy, (16, 4))
v = numpy.reshape(v, (4, 4))
print("v:\n", v)
print()
print("policy:\n", po)


# # Value Iteration

# In[68]:


v = [0]*16
num_states = 16
terminal = [0, 15]
reward = -1
while (True):
    delta = 0
    for state in range(num_states):
        val = v[state] 
        if (state in terminal):
            continue
        nval = -np.inf 
        for action in range(4):
            next_state = get_next_state(state, action)
            state_num = 4*next_state[0] + next_state[1]
            nval = max(nval, reward + v[state_num])
        v[state] = nval
        delta = max(delta, abs(val-v[state]))
    print("delta:" , delta)
    if delta < 0.00001:
        break

policy = [[0.25 for i in range(4)] for j in range(16)]
for state in range(num_states):
    if (state in terminal):
        continue
    old_action = []
    for act in policy[state]:
        old_action.append(act)

    li = []
    for action in range(4):
        next_state = get_next_state(state, action)
        next_state = 4*next_state[0] + next_state[1]
        li.append(reward + v[next_state])
    li = numpy.round(numpy.array(li), 2)
    maxval = max(li)
    count=0;indices=[]


    for action in range(4):
        if (li[action] == maxval):
            indices.append(action)
            count+=1;
    for index in range(4):
        if index in indices:
            policy[state][index] = 1/count;
        else:
            policy[state][index] = 0

po = numpy.reshape(policy, (16, 4))
v = numpy.reshape(v, (4, 4))
print(v)
print()
print(po)


# In[69]:


print(get_action_matrix(po))


# In[ ]:




