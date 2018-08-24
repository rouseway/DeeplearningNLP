#-*- coding:utf-8 -*-
#author: raosiwei
#date:2018-08-01

import numpy as np
import tensorflow as tf
from data_process import NNLMDataProcess
from nnlm import NNLanguageModelNetwork


input_path = ''
model_path = ''

#do statistic in corpus, filter low frequent word as <unk>,
#then count the vocabulary size
vocab_size = 10001

#network parameters
embedding_size = 128
rnn_size = 128
num_layers = 2
batch_size = 128
num_steps = 35
learning_rate = 1.0
max_grad_norm = 5
keep_prob = 0.9


def training():
    epochs = 20
    display_step = 10

    #deal corpus
    datas = NNLMDataProcess()
    datas.convert_sentences_to_number_sequences(input_path)
    datas.save_word_indexing_info(model_path)

    #construct NN-language Model
    nnlm = NNLanguageModelNetwork(vocab_size, embedding_size, rnn_size, num_layers, batch_size,
                                  num_steps, learning_rate, max_grad_norm, keep_prob)

    #do training
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        counter = 1
        for epoch_i in range(1, epochs+1):
            new_state = sess.run(nnlm.initial_state)
            for batch_i, (inputs, labels) in enumerate(datas.get_batches(batch_size, num_steps)):
                counter += 1
                _, loss = sess.run(
                    [nnlm.train_op, nnlm.cost],
                    feed_dict={
                        nnlm.inputs: inputs,
                        nnlm.targets: labels,
                        nnlm.initial_state: new_state
                    }
                )
                if counter % display_step == 0:
                    print("Epoch: {:>3}/{} Batch {:>4}/{}  -Training Loss: {:>6.3f}"
                          .format(epoch_i, epochs, batch_i, batch_size, loss))

        saver.save(sess, model_path+"nnlm.ckpt")
        print("NN language model saved successfully.")



def predicting():
    #start sentence
    start_sent = "i think"
    gen_text_lst = [ word for word in start_sent.split() ]
    max_gen_len = 20

    #load word indexing
    datas = NNLMDataProcess()
    datas.load_word_indexing_info(model_path)

    #choice most likely word
    def pick_top_n(predicts, v_size, top_n=5):
        p = np.squeeze(predicts)
        p[np.argsort(p)[:-top_n]] = 0
        p = p / np.sum(p)
        w = np.random.choice(v_size, 1, p=p)[0]
        return w

    #construct NN
    nnlm = NNLanguageModelNetwork(vocab_size, embedding_size, rnn_size, num_layers, batch_size,
                                  num_steps, learning_rate, max_grad_norm, keep_prob, training=False, online=True)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        #load model and do online predicting
        saver.restore(sess, model_path+"nnlm.ckpt")
        new_state = sess.run(nnlm.initial_state)
        for word in gen_text_lst:
            x = np.zeros((1, 1))
            x[0, 0] = datas.word2index[word]
            preds, new_state = sess.run(
                [nnlm.predict_out, nnlm.final_state],
                feed_dict={
                    nnlm.inputs: x,
                    nnlm.initial_state: new_state
                }
            )
        w = pick_top_n(preds, vocab_size)
        gen_text_lst.append(datas.index2word[w])

        for i in range(max_gen_len):
            x[0, 0] = w
            preds, new_state = sess.run(
                [nnlm.predict_out, nnlm.final_state],
                feed_dict={
                    nnlm.inputs: x,
                    nnlm.initial_state: new_state
                }
            )
            w = pick_top_n(preds, vocab_size)
            gen_text_lst.append(datas.index2word[w])

    print(' '.join(gen_text_lst))



if __name__ == "__main__":
    training()
    predicting()
