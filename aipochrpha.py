#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.keras
import numpy as np
import os
import time

def load_file(file_path):
        
    # parse text file:
    text = open(file_path, 'rb').read().decode(encoding='utf-8')
    print(f'Length of text: {len(text)}')
    
    # index vocabulary:
    vocab = sorted(set(text))
    print(f'Number of unique words: {len(vocab)}')
    
    # generate indexing dicts:
    char2idx = { u : i for i, u in enumerate(vocab) }
    idx2char = np.array(vocab)
    
    # vectorize text:
    text_vec = np.array([ char2idx[ch] for ch in text ])
    
    return vocab, text_vec, char2idx, idx2char

def construct_dataset(char_data, seq_len, batch_size, buffer_size=10000):
    char_dataset = tf.data.Dataset.from_tensor_slices(char_data)
    sequences = char_dataset.batch(seq_len+1,drop_remainder=True)
 
    seq_dataset = sequences.map(lambda s : (s[:-1], s[1:]))
    seq_dataset = seq_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
   
    return seq_dataset 

def construct_network(vocab_size, batch_size, embedding_dim=256, rnn_units=1024):
    network = tf.keras.Sequential([
        # append embedding layer:
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                            batch_input_shape=[batch_size, None]),
        
        # append recurrent layer:
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        # append decision layer:
        tf.keras.layers.Dense(vocab_size)
    ])
    
    return network;    

def network_loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True)

def main():
    
    
    file_in = 'shakespeare.txt' 
    seq_len = 100
    batch_size = 64
    n_epochs = 10
     
    # load dataset:
    vocab, data, char2idx, idx2char = load_file(file_in)
    
    # construct sequences from dataset:
    seq_dataset = construct_dataset(data, seq_len, batch_size)
     
    network = construct_network(len(vocab), batch_size)
    network.summary()

    # compile network:
    network.compile(optimizer='adam', loss=network_loss)
    
    history = network.fit(seq_dataset, epochs=n_epochs)

    print(seq_dataset)
    
if __name__ == '__main__':
    main()        
