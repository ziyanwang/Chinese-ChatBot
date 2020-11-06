#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
import operator
main_path = '/content/drive/My Drive/Colab Notebooks/'
question = np.load(main_path + 'middle_data/' + 'pad_question.npy')
answer = np.load(main_path + 'middle_data/' + 'pad_answer.npy')
answer_o = np.load(main_path + 'middle_data/' + 'answer_o.npy', allow_pickle=True)
with open(main_path + 'middle_data/' + 'vocab_bag.pkl', 'rb') as f:
    words = pickle.load(f)
with open(main_path + 'middle_data/' + 'pad_word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)
with open(main_path + 'middle_data/' + 'pad_index_to_word.pkl', 'rb') as f:
    index_to_word = pickle.load(f)
vocab_size = len(word_to_index) + 1
maxLen=20
def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
    return dir_list


# In[ ]:


from keras.preprocessing import sequence
def generate_train(batch_size):
    print('\n*********************************generate_train()*********************************')
    steps=0
    question_ = question
    answer_ = answer
    while True:
        batch_answer_o = answer_o[steps:steps+batch_size]
        batch_question = question_[steps:steps+batch_size]
        batch_answer = answer_[steps:steps+batch_size]
        outs = np.zeros([batch_size, maxLen, vocab_size], dtype='float32')
        for pos, i in enumerate(batch_answer_o):
            for pos_, j in enumerate(i):
                if pos_ > 20:
                    print(i)
                outs[pos, pos_, j] = 1 # one-hot
        yield [batch_question, batch_answer], outs
        steps += batch_size
        if steps == 100000:
            steps = 0


# In[ ]:


from keras.layers import Embedding
from keras.layers import Input, Dense, LSTM, TimeDistributed, Bidirectional, Dropout, Concatenate, RepeatVector, Activation, Dot
from keras.layers import concatenate, dot                    
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau
from keras.initializers import TruncatedNormal
import pydot
import os, re
truncatednormal = TruncatedNormal(mean=0.0, stddev=0.05)
embed_layer = Embedding(input_dim=vocab_size, 
                        output_dim=100, 
                        mask_zero=True,
                        input_length=None,
                        embeddings_initializer= truncatednormal)
# embed_layer.build((None,))

LSTM_encoder = LSTM(512,
                      return_sequences=True,
                      return_state=True,
#                       activation='relu',
#                       dropout=0.25,
#                       recurrent_dropout=0.1,
                      kernel_initializer= 'lecun_uniform',
                      name='encoder_lstm'
                        )
LSTM_decoder = LSTM(512, 
                    return_sequences=True, 
                    return_state=True, 
#                     activation = 'relu',
#                     dropout=0.25, 
#                     recurrent_dropout=0.1,
                    kernel_initializer= 'lecun_uniform',
                    name='decoder_lstm'
                   )

#encoder输入 与 decoder输入
input_question = Input(shape=(None, ), dtype='int32', name='input_question')
input_answer = Input(shape=(None, ), dtype='int32', name='input_answer')

input_question_embed = embed_layer(input_question)
input_answer_embed = embed_layer(input_answer)


encoder_lstm, question_h, question_c = LSTM_encoder(input_question_embed)

decoder_lstm, _, _ = LSTM_decoder(input_answer_embed, 
                                  initial_state=[question_h, question_c])

attention = dot([decoder_lstm, encoder_lstm], axes=[2, 2])
attention = Activation('softmax')(attention)
context = dot([attention, encoder_lstm], axes=[2,1])
decoder_combined_context = concatenate([context, decoder_lstm])

# output = dense1(decoder_combined_context)
# output = dense2(Dropout(0.5)(output))

# Has another weight + tanh layer as described in equation (5) of the paper
decoder_dense1 = TimeDistributed(Dense(256,activation="tanh"))
decoder_dense2 = TimeDistributed(Dense(vocab_size,activation="softmax"))
output = decoder_dense1(decoder_combined_context) # equation (5) of the paper
output = decoder_dense2(output) # equation (6) of the paper

model = Model([input_question, input_answer], output)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

filepath = main_path + "modles/W-" + "-{epoch:3d}-{loss:.4f}-.h5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=1,
                             save_weights_only=True
                             )
reduce_lr = ReduceLROnPlateau(monitor='loss', 
                              factor=0.2, 
                              patience=2, 
                              verbose=1, 
                              mode='min', 
                              min_delta=0.0001, 
                              cooldown=0, 
                              min_lr=0
                              )
tensorboard = TensorBoard(log_dir=main_path + 'logs', 
#                           histogram_freq=0, 
                          batch_size=100
#                           write_graph=True, 
#                           write_grads=True, 
#                           write_images=True, 
#                           embeddings_freq=0, 
#                           embeddings_layer_names=None, 
#                           embeddings_metadata=None, 
#                           embeddings_data=None, 
#                           update_freq='epoch'
                         )
callbacks_list = [checkpoint, reduce_lr, tensorboard]

initial_epoch_=0
file_list = os.listdir(main_path + 'modles/')
if len(file_list) > 0:
    epoch_list = get_file_list(main_path + 'modles/')
    epoch_last = epoch_list[-1]
    model.load_weights(main_path + 'modles/' + epoch_last)
    print("**********checkpoint_loaded: ", epoch_last)
    initial_epoch_ = int(epoch_last.split('-')[2]) - 1
    print('**********Begin from epoch: ', str(initial_epoch_))

model.fit_generator(generate_train(batch_size=100), 
                    steps_per_epoch=1000, # (total samples) / batch_size 100000/100 = 1000
                    epochs=200, 
                    verbose=1, 
                    callbacks=callbacks_list, 
#                     validation_data=generate_test(batch_size=100), 
#                     validation_steps=200, # 10000/100 = 100
                    class_weight=None, 
                    max_queue_size=5, 
                    workers=1, 
                    use_multiprocessing=False, 
                    shuffle=False, 
                    initial_epoch=initial_epoch_
                    )

model.summary()


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('pwd')
get_ipython().system('ls "/content/drive/My Drive/Colab Notebooks/modles/"')


# In[12]:


import  pickle
file_list = os.listdir(main_path + 'modles/')
loss_list = {}
for modelname in file_list:
    loss_list[int(modelname.split('-')[2])] = modelname.split('-')[3]
print(loss_list)
# np.save(main_path + 'middle_data/loss_list.npy', loss_list)
with open(main_path + 'middle_data/loss_list.npy', 'wb') as f:
    pickle.dump(loss_list, f, pickle.HIGHEST_PROTOCOL)

