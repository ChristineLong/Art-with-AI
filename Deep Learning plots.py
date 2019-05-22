#Import packages
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
#Sets the threshold for what messages will be logged.
old_v = tf.logging.get_verbosity()
# able to set the logging verbosity to either DEBUG, INFO, WARN, ERROR, or FATAL. Here its ERROR
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
#in the end
tf.logging.set_verbosity(old_v)


#import data and clean
import re

raw = pd.read_csv("wiki_movie_plots_deduped.csv")
scifi_raw = raw[raw['Genre'].str.contains("sci")]
scifi = scifi_raw.filter(["Plot"])

text = []

for i in range(len(scifi)):
    text += re.split('(\W)', scifi.iloc[i, 0])

while ("" in text):
    text.remove("")
while ("a" in text):
    text.remove("a")
while ("an" in text):
    text.remove("an")
while ("the" in text):
    text.remove("the")

corpus_raw = ''.join(text)

print("Corpus is {} characters long".format(len(corpus_raw)))



#Create functions
def create_lookup_tables(txt):
    """
    Create lookup tables for vocab
    :param text: The GOT text split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    vocab = set(txt)
    int_to_vocab = {key: word for key, word in enumerate(vocab)}
    vocab_to_int = {word: key for key, word in enumerate(vocab)}
    return vocab_to_int, int_to_vocab


def token_lookup():
    """
    Generate a dict to map punctuation into a token
    :return: dictionary mapping puncuation to token
    """
    return {
        '.': '||period||',
        ',': '||comma||',
        '"': '||quotes||',
        ';': '||semicolon||',
        '!': '||exclamation-mark||',
        '?': '||question-mark||',
        '(': '||left-parentheses||',
        ')': '||right-parentheses||',
        '--': '||emm-dash||',
        '\n': '||return||'

    }


import pickle

token_dict = token_lookup()
for token, replacement in token_dict.items():
    corpus_raw = corpus_raw.replace(token, ' {} '.format(replacement))
corpus_raw = corpus_raw.lower()
corpus_raw = corpus_raw.split()

vocab_to_int, int_to_vocab = create_lookup_tables(corpus_raw)
corpus_int = [vocab_to_int[word] for word in corpus_raw]
pickle.dump((corpus_int, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target data
    :param int_text: text with words replaced by their ids
    :param batch_size: the size that each batch of data should be
    :param seq_length: the length of each sequence
    :return: batches of data as a numpy array
    """
    words_per_batch = batch_size * seq_length
    num_batches = len(int_text) // words_per_batch
    int_text = int_text[:num_batches * words_per_batch]
    y = np.array(int_text[1:] + [int_text[0]])
    x = np.array(int_text)

    x_batches = np.split(x.reshape(batch_size, -1), num_batches, axis=1)
    y_batches = np.split(y.reshape(batch_size, -1), num_batches, axis=1)

    batch_data = list(zip(x_batches, y_batches))

    return np.array(batch_data)


num_epochs = 10000
batch_size = 512
rnn_size = 512
num_layers = 3
keep_prob = 0.7
embed_dim = 512
seq_length = 30
learning_rate = 0.001
save_dir = './save'

train_graph = tf.Graph()
with train_graph.as_default():
    # Initialize input placeholders
    input_text = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')

    # Calculate text attributes
    vocab_size = len(int_to_vocab)
    input_text_shape = tf.shape(input_text)

    # Build the RNN cell
    lstm = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size)
    drop_cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop_cell] * num_layers)

    # Set the initial state
    initial_state = cell.zero_state(input_text_shape[0], tf.float32)
    initial_state = tf.identity(initial_state, name='initial_state')

    # Create word embedding as input to RNN
    embed = tf.contrib.layers.embed_sequence(input_text, vocab_size, embed_dim)

    # Build RNN
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
    final_state = tf.identity(final_state, name='final_state')

    # Take RNN output and make logits
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)

    # Calculate the probability of generating each word
    probs = tf.nn.softmax(logits, name='probs')

    # Define loss function
    cost = tf.contrib.seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_text_shape[0], input_text_shape[1]])
    )

    # Learning rate optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Gradient clipping to avoid exploding gradients
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

import time

pickle.dump((seq_length, save_dir), open('params.p', 'wb'))
batches = get_batches(corpus_int, batch_size, seq_length)
num_batches = len(batches)
start_time = time.time()

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_index, (x, y) in enumerate(batches):
            feed_dict = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate
            }
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed_dict)

        time_elapsed = time.time() - start_time
        print(
            'Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}   time_elapsed = {:.3f}   time_remaining = {:.0f}'.format(
                epoch + 1,
                batch_index + 1,
                len(batches),
                train_loss,
                time_elapsed,
                ((num_batches * num_epochs) / ((epoch + 1) * (batch_index + 1))) * time_elapsed - time_elapsed))

        # save model every 10 epochs
        if epoch % 10 == 0:
            saver = tf.train.Saver()
            saver.save(sess, save_dir)
            print('Model Trained and Saved')

import pickle

corpus_int, vocab_to_int, int_to_vocab, token_dict = pickle.load(open('preprocess.p', mode='rb'))
seq_length, save_dir = pickle.load(open('params.p', mode='rb'))

import random


def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word with some randomness
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    return np.random.choice(list(int_to_vocab.values()), p=probabilities.ravel(), size=1)[0]


# Load the Graph and Generate
gen_length = 100
prime_words = 'in'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load the saved model
    loader = tf.train.import_meta_graph(save_dir + '.meta')
    loader.restore(sess, save_dir)

    # Get tensors from loaded graph
    input_text = loaded_graph.get_tensor_by_name('input:0')
    initial_state = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state = loaded_graph.get_tensor_by_name('final_state:0')
    probs = loaded_graph.get_tensor_by_name('probs:0')

    # Sentences generation setup
    gen_sentences = prime_words.split()
    prev_state = sess.run(initial_state, {input_text: np.array([[1 for word in gen_sentences]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])
        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        pred_word = pick_word(probabilities.ravel()[0:len(list(int_to_vocab))], int_to_vocab)
        gen_sentences.append(pred_word)

    # Remove tokens
    chapter_text = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        chapter_text = chapter_text.replace(' ' + token.lower(), key)

    print(chapter_text)

