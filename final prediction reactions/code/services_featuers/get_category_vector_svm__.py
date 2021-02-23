#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('config', 'Completer.use_jedi=False')
import pickle


# In[14]:


path="../models/SVC_category.pickle"
def load_model(path):
    with open(path,"rb") as file:
        return pickle.load(file)
SVC_category=load_model(path)
print(SVC_category)
with open("../models/TF_Vecotrize.pickle","rb") as file:
    TF_Vecorize=pickle.load(file)


# In[15]:


def predict_svm(clf,post,tfidf):
    post=tfidf.transform(post)
    return clf.predict_proba(post)


# In[19]:


def get_category_vector(post):
    return predict_svm(SVC_category,post,TF_Vecorize)
get_category_vector(["جديد قضيه ابن بايدن الاف بي حقق شريكه ساعات "])


# In[ ]:




