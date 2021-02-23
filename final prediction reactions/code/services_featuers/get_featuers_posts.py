#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')
from get_category_vector_svm__ import get_category_vector
from get_category_vector_nn import get_category_vector_nn
from get_emotion_vector import get_emotion_vector_svm
from get_emotion_vector_NN import get_emotion_vecor_nn
import numpy as np


# In[70]:


def concatentate_tow_numpyarray(a,b):
    k=list()
    for i in range(len(b)):
        k.append(a[i].tolist()+b[i].tolist())
    return np.array(k)
def get_vector_category_full(post):
    return concatentate_tow_numpyarray(get_category_vector(post),get_category_vector_nn(post))


# In[71]:


def get_vector_emotion_full(post):
    return concatentate_tow_numpyarray(get_emotion_vector_svm(post),get_emotion_vecor_nn(post))


# In[76]:


def convert_to_one_vector(nparray):
    l=list()
    for s in nparray:
        l=l+s.tolist()
    return np.array(l)
def get_featuers_for_post(post):
    return concatentate_tow_numpyarray(get_vector_category_full(post),get_vector_emotion_full(post))

