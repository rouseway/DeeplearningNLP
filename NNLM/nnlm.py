#-*- coding:utf-8 -*-
#author: raosiwei
#date: 2018-08-01


import tensorflow as tf


class NNLanguageModelNetwork(object):

    def __init__(self, vocab_size, embed_size, rnn_size, num_layers, batch_size,
                 num_steps, learning_rate, grad_clip, keep_prob, training=True, online=False):

        '''

        :param vocab_size:
        :param embed_size:
        :param rnn_size:
        :param num_layers:
        :param batch_size:
        :param num_steps:
        :param learning_rate:
        :param grad_clip:
        :param keep_prob:
        :param training:
        :param online:
        '''

        #set different data shapes in training
        if online == True:
            self.batch_size, self.num_steps = 1, 1
        else:
            self.batch_size, self.num_steps = batch_size, num_steps


        # reset tensorflow graph
        tf.reset_default_graph()


        #input layer
        self.inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='targets')


        #embedding layer
        embedding = tf.get_variable('embedding', [vocab_size, embed_size])
        embed_inputs = tf.nn.embedding_lookup(embedding, self.inputs)
        if training:
            embed_inputs = tf.nn.dropout(embed_inputs, self.keep_prob)


        #RNN layers
        dropout_keep_prob = keep_prob if training else 1.0
        lstm_cells = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(rnn_size),
                                          output_keep_prob=dropout_keep_prob) for _ in range(num_layers)
        ]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)


        #output layer
        outputs = []
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(embed_inputs[:, time_step, :], self.initial_state)
                outputs.append(cell_output)
                self.final_state = state
            rnn_output = tf.reshape(tf.concat(outputs, 1), [-1, rnn_size])

        weight = tf.get_variable('weight', [rnn_size, vocab_size])
        bias = tf.get_variable('bias', [vocab_size])
        logits = tf.matmul(rnn_output, weight) + bias
        self.predict_out = tf.nn.softmax(logits, name='predictions')


        #loss and optimizer
        with tf.name_scope('loss'):
            self.cost = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(self.targets, [-1]),
                logits=logits
            )) / batch_size
            trainable_variable = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variable), clip_norm=grad_clip)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, trainable_variable))
