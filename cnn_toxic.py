
from __future__ import print_function, division
from builtins import range

import os
import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.models import load_model
from sklearn.metrics import roc_auc_score

#configuration
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 200       #embedding dimenstion
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10

#load in pre-trained workd vectors
'''
print('Loading word vectors')

word2vec = {}

with open('D://software engineering/nlp_translate/glove.6B.200d.txt','r',encoding='utf-8') as f:
    for line in f:
        values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

#prepare text samples and labels
print('Loading in comments')

train = pd.read_csv("D:/software engineering/nlp_translate/train.csv")
sentences  =train["comment_text"].fillna("DUMMY_VALUE").values
possible_lables = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
targets = train[possible_lables].values

print("max sequence length:",max(len(s) for s in sentences))
print("min sequence length:", min(len(s)for s in sentences))
s=sorted(len(s) for s in sentences)
print("median sequence length:", s[len(s)//2])

#convert the sentences into integers
tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

word2idx = tokenizer.word_index
print('Found %s unique tokens.' %len(word2idx))

data = pad_sequences(sequences,maxlen = MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)

# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < MAX_VOCAB_SIZE:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector



# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False
)


print('Building model...')

# train a 1D convnet with global maxpooling
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(possible_lables), activation='sigmoid')(x)

model = Model(input_, output)
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

print('Training model...')
r = model.fit(data,targets,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_split=VALIDATION_SPLIT)


# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

# plot the mean AUC over each label
p = model.predict(data)
aucs = []
for j in range(6):
    auc = roc_auc_score(targets[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))

model.save('model1.h5')
'''
test = pd.read_csv("D:/software engineering/nlp_translate/test.csv")
print(test)

model = load_model('model1.h5')


print('test after load:', model.predict(test))