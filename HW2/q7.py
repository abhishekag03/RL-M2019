#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import numpy
import math
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


poisson_vals = {}
max_cars_a = 20
max_cars_b = 20
max_req_a = 10
max_req_b = 10
max_ret_a = 10
max_ret_b = 10
lamda_req_a = 3
lamda_req_b = 4
lamda_ret_a = 3 
lamda_ret_b = 2 


# In[7]:


math.exp(1)


# In[8]:


def get_poisson_val(lamda, n):
    global poisson_vals
    if ((lamda, n) in poisson_vals):
        return poisson_vals[(lamda, n)]
    else:
        poisson_vals[(lamda, n)] = (np.power(lamda, n) * np.exp(-lamda)) / np.math.factorial(n)
        return poisson_vals[(lamda, n)]


# In[9]:


def get_reward(sa, sb, v, action):
    carsA = max(min(sa - action, max_cars_a), 0)
    carsB = max(min(sb - action, max_cars_b), 0)    
    ret = 0
    for req_a in range(max_req_a):
        for req_b in range(max_req_b):
            for ret_a in range(max_ret_a):
                for ret_b in range(max_ret_b):
                    rent_a = min(carsA, req_a)
                    rent_b = min(carsB, req_b)
                    left_a = max(min(carsA - rent_a + ret_a, max_cars_a), 0)
                    left_b = max(min(carsB - rent_b + ret_b, max_cars_b), 0)
                    prob = get_poisson_val(lamda_req_a, req_a) * get_poisson_val(lamda_req_b, req_b) * get_poisson_val(lamda_ret_a, ret_a) * get_poisson_val(lamda_ret_b, ret_b)
                    reward = (rent_a + rent_b)*10
                    reward += abs(action) * (-2)
                    if (action > 0):
                        reward+=2
                    if (left_a > 10):
                        reward -=4
                    if (left_b > 10):
                        reward -=4
                    ret += prob * (reward + 0.9*v[int(left_a)][int(left_b)])
    return ret


# In[15]:


def policy_iteration():
    v = np.zeros((max_cars_a + 1, max_cars_b + 1))
    policy = np.zeros((max_cars_a + 1, max_cars_b + 1))
    num_states = (max_cars_a + 1) * (max_cars_b + 1)
    max_car_transfer = 5
    actions = []
    for i in range(-max_car_transfer, max_car_transfer + 1):
        actions.append(i)
    num_actions = len(actions)
    flag = True
    while (True):
        while (True):
            delta = 0
            for sa in range(max_cars_a + 1):
                for sb in range(max_cars_b + 1):
                    action = policy[sa][sb]
                    val = v[sa][sb]
                    reward = get_reward(sa, sb, v, action)
                    v[sa][sb] = reward
                    delta = max(delta, abs(val - reward))
            print("delta = ", delta)                    
            if delta < 1:
                break
        
    #     flag3 = True

        policy_stable = True
        for sa in range(max_cars_a + 1):
            for sb in range(max_cars_b + 1):
                old_action = policy[sa][sb]
                li = []
                for action in actions:
                    li.append(get_reward(sa, sb, v, action)) 
                    
                maxind = numpy.argmax(li)
                bestaction = actions[maxind]
                policy[sa][sb] = bestaction
                if (old_action != bestaction):
                    policy_stable = False      
        plt.pcolor(policy)
        plt.show()
        plot_3d(v)
        plt.show()
        if (policy_stable):
            break


# In[16]:


policy_iteration()


# In[14]:


#Plotting code taken from https://stackoverflow.com/questions/11766536/matplotlib-3d-surface-from-a-rectangular-array-of-heights
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
def plot_3d(v):
    X = np.arange(0, 21)
    Y = np.arange(0, 21)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, v, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


# In[ ]:




