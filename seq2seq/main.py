#-*- coding:utf-8 -*-
#author: raosiwei
#date: 2018-07-20

import tensorflow as tf
from data_process import Seq2SeqDataProcess
from seq2seq import Seq2SeqModel


source_path = ""
target_path = ""
model_dir = ""


def training():
    epochs = 100
    batch_size = 64
    rnn_size = 64
    num_layers = 2
    source_vocab_size = 5000
    target_vocab_size = 5000
    encoder_embed_size = 32
    decoder_embed_size = 32
    learning_rate = 0.001

    datas = Seq2SeqDataProcess()
    datas.convert_sentences_to_number_sequences(source_path, target_path)
    datas.save_word_mapping_info(model_dir)

    #construct graph
    train_graph = tf.Graph()
    with train_graph.as_default():
        #construct seq2seq model
        seq2seq = Seq2SeqModel(source_vocab_size, target_vocab_size, encoder_embed_size, decoder_embed_size,
                               rnn_size, num_layers, learning_rate, batch_size)

        training_decoder_output, predicting_decoder_output = seq2seq.seq2seq_model(datas.target_word2int['<GO>'],
                                                                                   datas.target_word2int['<EOS>'])

        training_logits = tf.identity(training_decoder_output.rnn_output, name='logits')
        predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')

        masks = tf.sequence_mask(seq2seq.target_sequence_length, seq2seq.max_target_sequence_length,
                                 dtype=tf.float32, name='masks')

        with tf.name_scope('optimization'):
            #loss function
            cost = tf.contrib.seq2seq.sequence_loss(training_logits, seq2seq.targets, masks)
            #optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)
            #Gradient clipping
            gradients = optimizer.compute_gradients(cost)
            capped_grandients = [ (tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None ]
            train_op = optimizer.apply_gradients(capped_grandients)


    #training data (you can use another method to prepare your validation data)
    train_source = datas.source_int_seq[batch_size:]
    train_target = datas.target_int_seq[batch_size:]
    valid_source = datas.source_int_seq[:batch_size]
    valid_target = datas.target_int_seq[:batch_size]
    (
        valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(
        datas.get_batches(valid_target, valid_source, batch_size)
    )


    display_step = 50
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(1, epochs+1):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                datas.get_batches(train_target, train_source, batch_size)
            ):
                _, loss = sess.run(
                    [train_op, cost],
                    feed_dict={
                        seq2seq.inputs: sources_batch,
                        seq2seq.targets: targets_batch,
                        seq2seq.learning_rate: learning_rate,
                        seq2seq.target_sequence_length: targets_lengths,
                        seq2seq.source_sequence_length: sources_lengths
                    }
                )
                if batch_i % display_step == 0:
                    validation_loss = sess.run(
                        [cost],
                        feed_dict={
                            seq2seq.inputs: valid_sources_batch,
                            seq2seq.targets: valid_targets_batch,
                            seq2seq.learning_rate: learning_rate,
                            seq2seq.target_sequence_length: valid_targets_lengths,
                            seq2seq.source_sequence_length: valid_sources_lengths,
                        }
                    )
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f} - Validation loss: {:>6.3f}'
                          .format(epoch_i, epochs, batch_i, len(train_source)//batch_size, loss, validation_loss[0]))

        #save model
        saver = tf.train.Saver()
        saver.save(sess, model_dir+"/seq2seq_model.ckpt")
        print("Model Trained and Saved.")




if __name__ == "__main__":
    training()
