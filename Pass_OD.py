#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os
import re
import numpy as np
import torch
from Veh import V
path = "C:/Users/w9714/Desktop/Nando lab/VRT-data/requests"
files= os.listdir(path) 
s = []
for file in files: 
          f = open(path+"/"+file); 
          iter_f = iter(f);
          str = ""
          for line in iter_f: 
              str = str + line
          s.append(str) 
a="".join(s)


# In[3]:


pattern = re.compile(r'(?<="lat": )\d+\.?\d*')
b=pattern.findall(a)
b_float = []
for num in b:
    b_float.append(float(num))
b_float


# In[4]:


pattern = re.compile(r'(?<="lon": -)\d+\.?\d*')
c=pattern.findall(a)
c_float = []
for num in c:
    c_float.append(float(num))
c_float


# In[5]:


b_around=np.around(b_float, decimals=3)
b_around.shape


# In[6]:


c_around=np.around(c_float, decimals=3)
c_around.shape


# In[7]:


b_ = np.reshape(b_around, (199902,1))
c_ = np.reshape(c_around, (199902,1))
stack=np.concatenate((c_,b_),axis=1)
stack.shape


# In[8]:


#seperate O$D
O_lon=[]
O_lat=[]
D_lon=[]
D_lat=[]
for i in range(199902):
    if i % 2 == 0:
        O_lon.append(stack[i,0])
        O_lat.append(stack[i,1])
for i in range(199902):
    if i % 2 == 1:
        D_lon.append(stack[i,0])
        D_lat.append(stack[i,1])


# In[9]:


stack


# In[10]:


print(max(O_lon),min(O_lon),max(O_lat),min(O_lat))


# In[11]:


print(max(D_lon),min(D_lon),max(D_lat),min(D_lat))


# In[12]:


#Origin
#all 99951
#read 1000 one time
x0=[]
x1=[]
for i in range(1000):
    x0.append((O_lon[i]-73.913)*1000)
    x1.append((O_lat[i]-40.701)*1000) 


# In[13]:


x0=np.around(x0)
x1=np.around(x1)
x0_= np.reshape(x0, (1000,1))
x1_= np.reshape(x1, (1000,1))
x=np.concatenate((x0_,x1_),axis=1)
x=np.int_(x)
x


# In[14]:


O=np.zeros((1000,200,200),dtype=int)
for i in range(1000):
    O[i,x[i,0],x[i,1]]=1


# In[15]:


#Destination
y0=[]
y1=[]
for i in range(1000):
    y0.append((D_lon[i]-min(D_lon))*1000)
    y1.append((D_lat[i]-min(D_lat))*1000)
y0=np.around(y0)
y1=np.around(y1)
y0_= np.reshape(y0, (1000,1))
y1_= np.reshape(y1, (1000,1))
y=np.concatenate((y0_,y1_),axis=1)
y=np.int_(y)
y


# In[16]:


D=np.zeros((1000,200,200),dtype=int)
for i in range(1000):
    D[i,y[i,0],y[i,1]]=1


# In[25]:


V.shape


# In[21]:


O.shape


# In[19]:


D.shape


# In[ ]:




