#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import load_model
import pickle
get_ipython().run_line_magic('config', 'Completer.use_jedi=False')
from keras.preprocessing.sequence import pad_sequences


# In[3]:


model_prediction_cnn=load_model('../models/CNN_category.h5')
model_prediction_lstm=load_model('../models/LSTM_category.h5')
with open('../models/tokenizer_NN.pickle', 'rb') as handle:
        tokenizer= pickle.load(handle)


# In[4]:


def predict_post_cnn(post):
#     post=tokenizer.transform()
    post=tokenizer.texts_to_sequences(post)
    post = pad_sequences(post, padding='post', maxlen=100)
    return model_prediction_cnn.predict(post)
# predict_post('جديد قضيه ابن بايدن الاف بي حقق شريكه ساعات')
def predict_post_lstm(post):
#     post=tokenizer.transform()
    post=tokenizer.texts_to_sequences(post)
    post = pad_sequences(post, padding='post', maxlen=100)
    return model_prediction_lstm.predict(post)


# In[5]:


def get_category_vector_cnn(post):
    return predict_post_cnn(post)
def get_ecategory_vector_lstm(post):
    return predict_post_lstm(post)


# In[8]:


def get_category_vector_nn(post):
    vector=get_category_vector_cnn(post)+get_ecategory_vector_lstm(post)
    vector=vector/2
    return vector

