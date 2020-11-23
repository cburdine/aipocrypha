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
parser.add_argument('-o', '--output', metavar='output', help='the output file')

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
        entropy=2.0,
        filter_map=None):

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
        predicted_char = idx2char[predicted_val]
        if filter_map is not None and predicted_char in filter_map.keys():
            predicted_char = filter_map[predicted_char]
        text_generated.append(predicted_char)

    # return result:
    return seed + ''.join(text_generated)

def write_output_file(output_file, text, seed=None):
    
    with open(output_file, 'w') as fout:
    
        # write seed to header:
        if seed is not None:
            fout.write('SEED:\n')
            fout.write(seed)
            fout.write('\n')
        
        # dump text content:
        fout.write(text)


def main():
    
    args = parser.parse_args()

    file_in = args.file
    file_out = args.output
    model_name = args.name
    seq_len = 1000
    batch_size = 64
    n_epochs = 10000
    output_len = 20000
     
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
    #history = network.fit(seq_dataset, epochs=n_epochs, callbacks=[checkpoint_callback])
    
    # rebuild model with weights to take one input at a time:
    network = construct_network(len(vocab), 1)
    network.load_weights(tf.train.latest_checkpoint(checkpoint_dir)) 
    network.build(tf.TensorShape([1,None]))

    # default filter map:
    filter_map = { str(i) : '' for i in range(10) }
    
    text_seed = 'In those days there was no king in Israel: every man did that which was right in his own eyes.'
    
    if file_out is not None:
        text = generate_text(network, seed=text_seed, 
                            char2idx=char2idx, 
                            idx2char=idx2char,
                            output_len=output_len,
                            filter_map=filter_map,
                            entropy=1.85)    
        
        write_output_file(file_out, text, seed=text_seed)

if __name__ == '__main__':
    main()        
