#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.keras
import numpy as np
import os
import time

import argparse

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

parser = argparse.ArgumentParser(
            description='Train a text generation model')

parser.add_argument('-f', '--file', metavar='file', help='file the model will be trained on')
parser.add_argument('-n', '--name', metavar='name', help='the name of the model')

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

def generate_text(model, seed, 
        char2idx, idx2char, output_len=100,
        entropy=1.0):

    # vectorize seed string:
    input_eval = [ char2idx[s] for s in seed ]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    model.reset_states()
    for i in range(output_len):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) / entropy
        predicted_val = \
            tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        
        # add newly generated character to model:
        input_eval = tf.expand_dims([predicted_val], 0)
        text_generated.append(idx2char[predicted_val])
    
    # return result:
    return seed + ''.join(text_generated)




def main():
    
    args = parser.parse_args()

    file_in = args.file
    model_name = args.name
    seq_len = 100
    batch_size = 64
    n_epochs = 10
     
    # load dataset:
    vocab, data, char2idx, idx2char = load_file(file_in)
    
    # construct sequences from dataset:
    seq_dataset = construct_dataset(data, seq_len, batch_size)
    
    # build network: 
    network = construct_network(len(vocab), batch_size)
    network.summary()
    
    # compile network:
    network.compile(optimizer='adam', loss=network_loss)
    
    # add checkpoint directory:
    checkpoint_dir = f'./{model_name}'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)     

    # train model:
    history = network.fit(seq_dataset, epochs=n_epochs, callbacks=[checkpoint_callback])
    
    # rebuild model with weights to take one input at a time:
    network = construct_network(len(vocab), 1)
    network.load_weights(tf.train.latest_checkpoint(checkpoint_dir)) 
    network.build(tf.TensorShape([1,None]))
    
    text_seed = 'In those days there was no king in Israel: every man did that which was right in his own eyes.'

    print(generate_text(network, seed=text_seed, 
                        char2idx=char2idx, 
                        idx2char=idx2char))
    
if __name__ == '__main__':
    main()        
