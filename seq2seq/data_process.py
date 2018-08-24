#-*- coding:utf-8 -*-
#author: raosiwei
#date: 2018-07-20

import pickle
import numpy as np


class Seq2SeqDataProcess(object):

    def __init__(self):
        #word-number mapping index
        self.source_int2word = None
        self.source_word2int = None
        self.target_int2word = None
        self.target_word2int = None

        #sentences' number expression
        self.source_int_seq = None
        self.target_int_seq = None

        #special words
        self.special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']


    def extract_word_vocab(self, lines):
        doc_words = list(set([ word for line in lines for word in line.strip().split() ]))
        #merge special words and doc words into gram-number mapping index
        int2vocab = { idx : word for idx, word in enumerate(self.special_words + doc_words) }
        vocab2int = { word : idx for idx, word in int2vocab.items() }
        return int2vocab, vocab2int


    def convert_sentences_to_number_sequences(self, source_file, target_file):
        '''
        Read source file and target file, whose each line is a sentence. One sentence at a certain line in source file
        is corresponding to target file's same line.
        :param source_file: source file path (file is encoded with UTF-8)
        :param target_file: target file path (file is encoded with UTF-8)
        :return: null
        '''
        with open(source_file, 'r', encoding='utf-8') as fin:
            source_data = fin.read().split('\n')
        with open(target_file, 'r', encoding='utf-8') as fin:
            target_data = fin.read().split('\n')

        self.source_int2word, self.source_word2int = self.extract_word_vocab(source_data)
        self.target_int2word, self.target_word2int = self.extract_word_vocab(target_data)

        self.source_int_seq = [ [ self.source_word2int.get(word, self.source_word2int['<UNK>'])
                                  for word in line ] for line in source_data ]
        self.target_int_seq = [ [ self.target_word2int.get(word, self.target_word2int['<UNK>'])
                                  for word in line ] + [self.target_word2int['<EOS>']] for line in target_data ]


    def convert_source_sentence_to_number_sequence(self, sentence, max_seq_length):
        word_list = sentence.strip().split()
        return [ self.source_word2int.get(word, self.source_word2int['<UNK>']) for word in word_list \
                 + [self.source_word2int['<PAD>']] * (max_seq_length-len(word_list)) ]


    def save_word_mapping_info(self, save_dir):
        src_out = open(save_dir + "source.map", 'wb')
        pickle.dump(self.source_word2int, src_out)
        src_out.close()

        tgt_out = open(save_dir + "target.map", 'wb')
        pickle.dump(self.target_word2int, tgt_out)
        tgt_out.close()


    def load_word_mapping_info(self, save_dir):
        src_in = open(save_dir + "source.map", 'rb')
        self.source_word2int = pickle.load(src_in)
        tgt_in = open(save_dir + "target.map", 'rb')
        self.target_word2int = pickle.load(tgt_in)


    def pad_sentence_batch(self, sentence_batch, pad_int, max_sent_len=None):
        if max_sent_len == None:
            max_sent_len = max([ len(sentence) for sentence in sentence_batch ])
            return [ sentence + [pad_int] * (max_sent_len - len(sentence)) for sentence in sentence_batch ]
        else:
            ret = []
            for sentence in sentence_batch:
                if len(sentence) < max_sent_len:
                    ret.append(sentence + [pad_int] * (max_sent_len-len(sentence)))
                else:
                    ret.append(sentence[:max_sent_len])
            return ret


    def get_batches(self, targets, sources, batch_size, max_sent_len=None):
        for batch_i in range(0, len(sources)//batch_size):
            start_i = batch_i * batch_size
            sources_batch = sources[start_i:start_i + batch_size]
            targets_batch = targets[start_i:start_i + batch_size]
            #computing the sequence
            pad_sources_batch = np.array(self.pad_sentence_batch(sources_batch, self.source_word2int['<PAD>'], max_sent_len))
            pad_targets_batch = np.array(self.pad_sentence_batch(targets_batch, self.target_word2int['<PAD>'], max_sent_len))

            targets_lengths = []
            for target in targets_batch:
                targets_lengths.append(len(target))
            sources_lengths = []
            for source in sources_batch:
                sources_lengths.append(len(source))

            yield pad_targets_batch, pad_sources_batch, targets_lengths, sources_lengths
