#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
a = np.loadtxt('real_location_matrix.txt')
a_around=np.around(a, decimals=3)
a_around


# In[3]:


v_lon=[]
v_lat=[]
for i in range(1718):
    v_lon.append(a_around[i,0])
    v_lat.append(a_around[i,1])
v_lat[0]


# In[4]:


x0=[]
x1=[]
for i in range(1718):
    x0.append((v_lon[i]-73.911)*1000)
    x1.append((v_lat[i]-40.701)*1000)
x0=np.around(x0)
x1=np.around(x1)
x0_= np.reshape(x0, (1718,1))
x1_= np.reshape(x1, (1718,1))
x=np.concatenate((x0_,x1_),axis=1)
x=np.int_(x)
x


# In[8]:


V=np.zeros((1000,200,200),dtype=int)#1718,200,200
for i in range(1000):
    V[i,x[i,0],x[i,1]]=1


# In[10]:


V.shape


# In[ ]:




