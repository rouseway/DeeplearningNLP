#-*- coding:utf-8 -*-
#author: raosiwei
#date: 2018-08-01

import pickle
import numpy as np


class NNLMDataProcess(object):

    def __init__(self):
        #word-id mapping
        self.word2index = None
        self.index2word = None
        #sentences' number express
        self.sentence_int_express = None


    def extract_word_vocab(self, data_lines):
        special_words = ['<unk>', '<eos>']
        doc_words = list(set([ word for line in data_lines for word in line.strip().split() ]))
        #merge special words and doc words into vocab and number them
        int2vocab = {idx: word for idx, word in enumerate(list(set(special_words + doc_words)))}
        vocab2int = {word: idx for idx, word in int2vocab.items()}
        return int2vocab, vocab2int


    def convert_sentences_to_number_sequences(self, input_path, data_lines=None):
        if data_lines == None:
            with open(input_path, 'r', encoding='utf-8') as fin:
                data_lines = fin.readlines()
        self.index2word, self.word2index = self.extract_word_vocab(data_lines)
        self.sentence_int_express = [ [ self.word2index.get(word, self.word2index['<unk>'])
                                       for word in line.strip().split() ] for line in data_lines ]


    def save_word_indexing_info(self, save_dir):
        fout = open(save_dir + "/word-indexing.map", 'wb')
        pickle.dump(self.word2index, fout)
        fout.close()
        fout = open(save_dir + "/indexing-word.map", 'wb')
        pickle.dump(self.index2word, fout)
        fout.close()


    def load_word_indexing_info(self, save_dir):
        fin = open(save_dir + "/word-indexing.map", 'wb')
        self.word2index = pickle.load(fin)
        fin.close()
        fin = open(save_dir + "/indexing-word.map", 'wb')
        self.index2word = pickle.load(fin)
        fin.close()


    def get_batches(self, batch_size, sequence_length):
        flat_inputs = [ word for line in self.sentence_int_express for word in line ]
        for batch_i in range(0, len(flat_inputs)//(batch_size*sequence_length)):
            batch_start = batch_i * batch_size * sequence_length
            input_batch = []
            label_batch = []
            for step_i in range(0, batch_size):
                step_start = batch_start + step_i*sequence_length
                input_batch.append(np.array(flat_inputs[step_start: step_start+sequence_length], dtype=np.int32))
                label_batch.append(np.array(flat_inputs[step_start+1: step_start+sequence_length+1], dtype=np.int32))
            yield input_batch, label_batch
