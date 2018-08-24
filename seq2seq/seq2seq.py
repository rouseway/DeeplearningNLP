#-*- coding:utf-8 -*-
#author: raosiwei
#date: 2018-07-20

import tensorflow as tf
from tensorflow.python.layers.core import Dense


class Seq2SeqModel(object):

    def __init__(self, source_vocab_size, target_vocab_size, encoder_embed_size, decoder_embed_size, rnn_size,
                 num_layers, learning_rate, batch_size):

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.encoder_embed_size = encoder_embed_size
        self.decoder_embed_size = decoder_embed_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        #placeholder of the network
        self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        self.max_target_sequence_length = tf.reduce_max(self.target_sequence_length, name='max_target_length')
        self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')


    def get_lstm_cell(self, rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell


    def get_encoder_layer(self):
        #Encoder embedding
        encoder_embed_input = tf.contrib.layers.embed_sequence(self.inputs, self.source_vocab_size, self.encoder_embed_size)

        #RNN cell
        cell = tf.contrib.rnn.MultiRNNCell([self.get_lstm_cell(self.rnn_size) for _ in range(self.num_layers)])
        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                          sequence_length=self.source_sequence_length, dtype=tf.float32)
        return encoder_output, encoder_state


    def process_decoder_input(self, go_tag):
        ending = tf.strided_slice(self.targets, [0, 0], [self.batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], go_tag), ending], 1)
        return decoder_input


    def get_decoder_layer(self, encoder_state, decoder_input,  go_tag, eos_tag):
        #Decoder embedding
        decoder_embeddings = tf.Variable(tf.random_uniform([self.target_vocab_size, self.decoder_embed_size]))
        decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

        #RNN cell
        cell = tf.contrib.rnn.MultiRNNCell([self.get_lstm_cell(self.rnn_size) for _ in range(self.num_layers)])

        #Output layer
        output_layer = Dense(self.target_vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        #Training decoder
        with tf.variable_scope("decode"):
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                                sequence_length=self.target_sequence_length,
                                                                time_major=False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell, training_helper, encoder_state, output_layer)
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                              impute_finished=True,
                                                                              maximum_iteration=self.max_target_sequence_length)

        #Predicting decoder
        with tf.variable_scope("decode", reuse=True):
            start_tokens = tf.tile(tf.constant([go_tag], dtype=tf.int32), [self.batch_size], name='start_tokens')
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, start_tokens, eos_tag)
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell, predicting_helper, encoder_state, output_layer)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                                impute_finished=True,
                                                                                maximum_iteration=self.max_target_sequence_length)

        return training_decoder_output, predicting_decoder_output



    
    def seq2seq_model(self, go_tag, eos_tag):

        #get the encoder's state output
        _, encoder_state = self.get_encoder_layer()

        #decoder's input
        decoder_input = self.process_decoder_input(go_tag)

        #put state-output and decoder-input into decoder
        return self.get_decoder_layer(encoder_state, decoder_input, go_tag, eos_tag)

