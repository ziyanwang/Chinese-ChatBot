#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import operator
import pandas as pd
import jieba
from language.langconv import *


# In[4]:


def Traditional2Simplified(sentence):
    sentence = Converter('zh-hans').convert(sentence)
    return sentence
def is_all_chinese(strs):
    for chart in strs:
        if chart < u'\u4e00' or chart > u'\u9fff':
            return False
    return True
with open('qingyun.tsv', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
    lines = lines[:-2]
question = []
answer = []
for pos, line in enumerate(lines):
    if '\t' not in line:
        print(line)
    line = line.split('\t')
    q = line[0].strip()
    a = line[1].strip()
    question.append(' '.join(jieba.lcut(Traditional2Simplified(q).strip(), cut_all=False)))
    answer.append(' '.join(jieba.lcut(Traditional2Simplified(a).strip(), cut_all=False)))

print(len(question))
print(answer[:10])


# In[5]:


def is_all_chinese(strs):
    for chart in strs:
        if chart < u'\u4e00' or chart > u'\u9fff':
            return False
    return True
character = set()
for seq in question + answer:
    word_list = seq.split(' ')
    for word in word_list:
        if not is_all_chinese(word):
            character.add(word)
def is_pure_english(keyword):  
    return all(ord(c) < 128 for c in keyword)
character=list(character)
stop_words = set()
for pos, word in enumerate(character):
    if not is_pure_english(word):
        stop_words.add(word)
print('stop_words: ', len(stop_words))


# In[6]:


maxLen=18
for pos, seq in enumerate(question):
    seq_list = seq.split(' ')
    for epoch in range(3):
        for pos_, word in enumerate(seq_list):
            if word in stop_words:
                seq_list.pop(pos_)
    if len(seq_list) > maxLen:
        seq_list = seq_list[:maxLen]
    question[pos] = ' '.join(seq_list)
for pos, seq in enumerate(answer):
    seq_list = seq.split(' ')
    for epoch in range(3):
        for pos_, word in enumerate(seq_list):
            if word in stop_words:
                seq_list.pop(pos_)
    if len(seq_list) > maxLen:
        seq_list = seq_list[:maxLen]
    answer[pos] = ' '.join(seq_list)
    
answer_a = ['BOS ' + i + ' EOS' for i in answer]
answer_b = [i + ' EOS' for i in answer]
print(question[:10])
print(answer_a[:10])
print(answer_b[:10])


# In[7]:


import  pickle
counts = {}
BE = ['BOS', 'EOS']
for word_list in question + answer + BE:
    for word in word_list.split(' '):
        counts[word] = counts.get(word, 0) + 1 
word_to_index = {}
for pos, i in enumerate(counts.keys()):
    word_to_index[i] = pos
    
index_to_word = {}
for pos, i in enumerate(counts.keys()):
    index_to_word[pos] = i
    
vocab_bag =list(word_to_index.keys())
with open('word_to_index.pkl', 'wb') as f:
    pickle.dump(word_to_index, f, pickle.HIGHEST_PROTOCOL)
with open('index_to_word.pkl', 'wb') as f:
    pickle.dump(index_to_word, f, pickle.HIGHEST_PROTOCOL)
with open('vocab_bag.pkl', 'wb') as f:
    pickle.dump(vocab_bag, f, pickle.HIGHEST_PROTOCOL)
print('vocab_bag: ', len(vocab_bag))


# In[8]:


question = np.array([[word_to_index[w] for w in i.split(' ')] for i in question])
answer_a = np.array([[word_to_index[w] for w in i.split(' ')] for i in answer_a])
answer_b = np.array([[word_to_index[w] for w in i.split(' ')] for i in answer_b])
print(question[:3])
print(answer_a[:3])
print(answer_b[:3])


# In[9]:


import os 
import numpy as np
print('question: ', len(question), '\n', 'answer: ', len(answer))
np.save('question.npy', question[:100000])
np.save('answer_a.npy', answer_a[:100000])
np.save('answer_b.npy', answer_b[:100000])
print('Done!')


# In[10]:


import numpy as np
import pickle
import operator

question = np.load('question.npy')
answer_a = np.load('answer_a.npy')
answer_b = np.load('answer_b.npy')
print('answer_a.shape: ', answer_a.shape)
with open('word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)

for i, j in word_to_index.items():
    word_to_index[i] = j + 1

index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key
pad_question = question
pad_answer_a = answer_a
pad_answer_b = answer_b
maxLen = 20
for pos, i in enumerate(pad_question):
    for pos_, j in enumerate(i):
        i[pos_] = j + 1
    if(len(i) > maxLen):
        pad_question[pos] = i[:maxLen]
    
for pos, i in enumerate(pad_answer_a):
    for pos_, j in enumerate(i):
        i[pos_] = j + 1
    if(len(i) > maxLen):
        pad_answer_a[pos] = i[:maxLen]
for pos, i in enumerate(pad_answer_b):
    for pos_, j in enumerate(i):
        i[pos_] = j + 1
    if(len(i) > maxLen):
        pad_answer_b[pos] = i[:maxLen]
np.save('answer_o.npy', pad_answer_b)        
    
with open('vocab_bag.pkl', 'rb') as f:
    words = pickle.load(f)
vocab_size = len(word_to_index) + 1
print('word_to_vec_map: ', len(list(words)))
print('vocab_size: ', vocab_size)


from keras.preprocessing import sequence
#后端padding
pad_question = sequence.pad_sequences(pad_question, maxlen=maxLen,
                                      dtype='int32', padding='post', 
                                       truncating='post')
pad_answer = sequence.pad_sequences(pad_answer_a, maxlen=maxLen,
                                 dtype='int32', padding='post',
                                 truncating='post')

def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
    return dir_list

with open('pad_word_to_index.pkl', 'wb') as f:
    pickle.dump(word_to_index, f, pickle.HIGHEST_PROTOCOL)
with open('pad_index_to_word.pkl', 'wb') as f:
    pickle.dump(index_to_word, f, pickle.HIGHEST_PROTOCOL)
np.save('pad_question.npy', pad_question)
np.save('pad_answer.npy', pad_answer)
    
print(pad_answer[:3])
print(pad_answer_b[:3])


# In[ ]:




