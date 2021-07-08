import tensorflow as tf
import numpy as np

from konlpy.tag import Twitter
import pandas as pd
import enum
import os
import re
import json
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

from preprocess import *
import urllib.request

import sys
sys.path.append('../models')
from transformer import *
#from transformer import Transformer

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

def loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

def accuracy(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.expand_dims(tf.cast(mask, dtype=pred.dtype), axis=-1)
    pred *= mask    
    acc = train_accuracy(real, pred)

    return tf.reduce_mean(acc)


DATA_IN_PATH = './'
DATA_OUT_PATH = './'
TRAIN_INPUTS = 'train_inputs.npy'
TRAIN_OUTPUTS = 'train_outputs.npy'
TRAIN_TARGETS = 'train_targets.npy'
DATA_CONFIGS = 'data_configs.json'

SEED_NUM = 1234
tf.random.set_seed(SEED_NUM)

index_inputs = np.load(open(DATA_IN_PATH + TRAIN_INPUTS, 'rb'))
index_outputs = np.load(open(DATA_IN_PATH + TRAIN_OUTPUTS , 'rb'))
index_targets = np.load(open(DATA_IN_PATH + TRAIN_TARGETS , 'rb'))
prepro_configs = json.load(open(DATA_IN_PATH + DATA_CONFIGS, 'r'))

char2idx = prepro_configs['char2idx']
end_index = prepro_configs['end_symbol']
model_name = 'transformer'
vocab_size = prepro_configs['vocab_size']
#BATCH_SIZE = 2
BATCH_SIZE = 64
MAX_SEQUENCE = 25
EPOCHS = 30
VALID_SPLIT = 0.1

kargs = {'model_name': model_name,
         'num_layers': 2,
         'd_model': 512,
         'num_heads': 8,
         'dff': 2048,
         'input_vocab_size': vocab_size,
         'target_vocab_size': vocab_size,
         'maximum_position_encoding': MAX_SEQUENCE,
         'end_token_idx': char2idx[end_index],
         'rate': 0.1
        }

model = Transformer(**kargs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=loss,
              metrics=[accuracy])

checkpoint_path = './transformer_weights.h5.tmp'
earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10)
cp_callback = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, 
    save_best_only=True, save_weights_only=True)

#history = model.fit([index_inputs, index_outputs], index_targets, 
#                    batch_size=BATCH_SIZE, epochs=1,
#                    validation_split=VALID_SPLIT, callbacks=[earlystop_callback, cp_callback])

# need training for sample
sample_inputs = index_inputs[:10,:]
sample_outputs = index_outputs[:10,:]
sample_targets = index_targets[:10,:]
model.train_on_batch([sample_inputs, sample_outputs], sample_targets)

# 모델 불러오기
DATA_OUT_PATH = './'
SAVE_FILE_NM = 'transformer_weights.h5'


#model.built = True
#model.load_weights(os.path.join(DATA_OUT_PATH, model_name, SAVE_FILE_NM))
model.load_weights(os.path.join(DATA_OUT_PATH, SAVE_FILE_NM))

char2idx = prepro_configs['char2idx']
idx2char = prepro_configs['idx2char']

text = "남자친구 승진 선물로 뭐가 좋을까?"
test_index_inputs, _ = enc_processing([text], char2idx)
outputs = model.inference(test_index_inputs)

print('output:')
print(' '.join([idx2char[str(o)] for o in outputs]))

