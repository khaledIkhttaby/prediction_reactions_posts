#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pickle
import numpy as np
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[28]:


paths=["../models/SVR_love.pickle","../models/SVR_wow.pickle","../models/SVR_angry.pickle","../models/SVR_sad.pickle","../models/SVR_care.pickle"]
emotion=["love","wow","angry","sad","care"]

def load_models():
    models={}
    
    for i in range(len(paths)):
        with open(paths[i],"rb") as file:
            models[emotion[i]]=pickle.load(file)
   
        
    with open("../models/TF_Vecotrize.pickle","rb") as file:
        TF_Vecorize=pickle.load(file)
    return models,TF_Vecorize
models,TF_Vectorize=load_models()


# In[31]:


def get_emotion_vector_svm(post):
    X=TF_Vectorize.transform(post)

    Emotion=[models['love'].predict(X),models['wow'].predict(X),models['angry'].predict(X),models['sad'].predict(X),models['care'].predict(X)]
    Emotion=np.array(Emotion).transpose()
    return Emotion/np.sum(Emotion,axis=1)[:,None]
#              /np.sum(Emotion)


# In[32]:


E=get_emotion_vector_svm(["جديد قضية ابن بايدن الأف بي آي حقق مع شريكه ","جديد قضية ابن بايدن الأف بي آي حقق مع شريكه "])




