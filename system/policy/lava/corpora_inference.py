from __future__ import unicode_literals

import json
import logging
from collections import Counter

import numpy as np

from system.policy.lava.latent_dialog.utils import Pack

PAD = '<pad>'
UNK = '<unk>'
USR = 'YOU:'
SYS = 'THEM:'
BOD = '<d>'
EOD = '</d>'
BOS = '<s>'
EOS = '<eos>'
SEL = '<selection>'
SPECIAL_TOKENS_DEAL = [PAD, UNK, USR, SYS, BOD, EOS]
SPECIAL_TOKENS = [PAD, UNK, USR, SYS, BOS, BOD, EOS, EOD]
STOP_TOKENS = [EOS, SEL]
DECODING_MASKED_TOKENS = [PAD, UNK, USR, SYS, BOD]
REQ_TOKENS = {}
DOMAIN_REQ_TOKEN = ['restaurant',  'hospital', 'hotel','attraction', 'train', 'police', 'taxi']
ACTIVE_BS_IDX = [13, 30, 35, 61, 72, 91, 93] #indexes in the BS indicating if domain is active
NO_MATCH_DB_IDX = [-1, 0, -1, 6, 12, 18, -1] # indexes in DB pointer indicating 0 match is found for that domain, -1 mean that domain has no DB
REQ_TOKENS['attraction'] = ["[attraction_address]", "[attraction_name]", "[attraction_phone]", "[attraction_postcode]", "[attraction_reference]", "[attraction_type]"]
REQ_TOKENS['hospital'] = ["[hospital_address]", "[hospital_department]", "[hospital_name]", "[hospital_phone]", "[hospital_postcode]"] #, "[hospital_reference]"
REQ_TOKENS['hotel'] = ["[hotel_address]", "[hotel_name]", "[hotel_phone]", "[hotel_postcode]", "[hotel_reference]", "[hotel_type]"]
REQ_TOKENS['restaurant'] = ["[restaurant_name]", "[restaurant_address]", "[restaurant_phone]", "[restaurant_postcode]", "[restaurant_reference]"]
REQ_TOKENS['train'] = ["[train_id]", "[train_reference]"]
REQ_TOKENS['police'] = ["[police_address]", "[police_phone]", "[police_postcode]"] #"[police_name]", 
REQ_TOKENS['taxi'] = ["[taxi_phone]", "[taxi_type]"]



class NormMultiWozCorpus(object):
    logger = logging.getLogger()

    def __init__(self, train_data_path, max_vocab_size):
        self.bs_size = 94
        self.db_size = 30
        self.bs_types = ['b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'b', 'b', 'c', 'c', 'c', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c',
                         'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'b', 'b', 'b']
        self.domains = ['hotel', 'restaurant', 'train',
                        'attraction', 'hospital', 'police', 'taxi']
        self.info_types = ['book', 'fail_book', 'fail_info', 'info', 'reqt']
        self.max_vocab_size = max_vocab_size
        self.tokenize = lambda x: x.split()
        self.train_corpus = self._read_file(train_data_path)
        self._extract_vocab()
        self._extract_goal_vocab()
        self.logger.info('Loading corpus finished.')

    def _read_file(self, train_data_path):
        train_data = json.load(open(train_data_path))
        train_data = self._process_dialogue(train_data)

        return train_data

    def _process_dialogue(self, data):
        new_dlgs = []
        all_sent_lens = []
        all_dlg_lens = []

        for key, raw_dlg in data.items():
            norm_dlg = [Pack(speaker=USR, utt=[BOS, BOD, EOS], bs=[
                             0.0]*self.bs_size, db=[0.0]*self.db_size)]
            for t_id in range(len(raw_dlg['db'])):
                usr_utt = [BOS] + self.tokenize(raw_dlg['usr'][t_id]) + [EOS]
                sys_utt = [BOS] + self.tokenize(raw_dlg['sys'][t_id]) + [EOS]
                norm_dlg.append(Pack(speaker=USR, utt=usr_utt,
                                     db=raw_dlg['db'][t_id], bs=raw_dlg['bs'][t_id]))
                norm_dlg.append(Pack(speaker=SYS, utt=sys_utt,
                                     db=raw_dlg['db'][t_id], bs=raw_dlg['bs'][t_id]))
                all_sent_lens.extend([len(usr_utt), len(sys_utt)])
            # To stop dialog
            norm_dlg.append(Pack(speaker=USR, utt=[BOS, EOD, EOS], bs=[
                            0.0]*self.bs_size, db=[0.0]*self.db_size))
            # if self.config.to_learn == 'usr':
            #     norm_dlg.append(Pack(speaker=USR, utt=[BOS, EOD, EOS], bs=[0.0]*self.bs_size, db=[0.0]*self.db_size))
            all_dlg_lens.append(len(raw_dlg['db']))
            processed_goal = self._process_goal(raw_dlg['goal'])
            new_dlgs.append(Pack(dlg=norm_dlg, goal=processed_goal, key=key))

        self.logger.info('Max utt len = %d, mean utt len = %.2f' % (
            np.max(all_sent_lens), float(np.mean(all_sent_lens))))
        self.logger.info('Max dlg len = %d, mean dlg len = %.2f' % (
            np.max(all_dlg_lens), float(np.mean(all_dlg_lens))))
        return new_dlgs

    def _extract_vocab(self):
        all_words = []
        for dlg in self.train_corpus:
            for turn in dlg.dlg:
                all_words.extend(turn.utt)
        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        keep_vocab_size = min(self.max_vocab_size, raw_vocab_size)
        oov_rate = np.sum(
            [c for t, c in vocab_count[0:keep_vocab_size]]) / float(len(all_words))

        self.logger.info('cut off at word {} with frequency={},\n'.format(vocab_count[keep_vocab_size - 1][0],
                                                                          vocab_count[keep_vocab_size - 1][1]) +
                         'OOV rate = {:.2f}%'.format(100.0 - oov_rate * 100))

        vocab_count = vocab_count[0:keep_vocab_size]
        self.vocab = SPECIAL_TOKENS + \
            [t for t, cnt in vocab_count if t not in SPECIAL_TOKENS]
        self.vocab_dict = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.vocab_dict[UNK]
        self.logger.info("Raw vocab size {} in train set and final vocab size {}".format(
            raw_vocab_size, len(self.vocab)))

    def _process_goal(self, raw_goal):
        res = {}
        for domain in self.domains:
            all_words = []
            d_goal = raw_goal[domain]
            if d_goal:
                for info_type in self.info_types:
                    sv_info = d_goal.get(info_type, dict())
                    if info_type == 'reqt' and isinstance(sv_info, list):
                        all_words.extend(
                            [info_type + '|' + item for item in sv_info])
                    elif isinstance(sv_info, dict):
                        all_words.extend(
                            [info_type + '|' + k + '|' + str(v) for k, v in sv_info.items()])
                    else:
                        print('Fatal Error!')
                        exit(-1)
            res[domain] = all_words
        return res

    def _extract_goal_vocab(self):
        self.goal_vocab, self.goal_vocab_dict, self.goal_unk_id = {}, {}, {}
        for domain in self.domains:
            all_words = []
            for dlg in self.train_corpus:
                all_words.extend(dlg.goal[domain])
            vocab_count = Counter(all_words).most_common()
            raw_vocab_size = len(vocab_count)
            discard_wc = np.sum([c for t, c in vocab_count])

            self.logger.info('================= domain = {}, \n'.format(domain) +
                             'goal vocab size of train set = %d, \n' % (raw_vocab_size,) +
                             'cut off at word %s with frequency = %d, \n' % (vocab_count[-1][0], vocab_count[-1][1]) +
                             'OOV rate = %.2f' % (1 - float(discard_wc) / len(all_words),))

            self.goal_vocab[domain] = [UNK] + [g for g, cnt in vocab_count]
            self.goal_vocab_dict[domain] = {
                t: idx for idx, t in enumerate(self.goal_vocab[domain])}
            self.goal_unk_id[domain] = self.goal_vocab_dict[domain][UNK]

    def get_corpus(self):
        id_train = self._to_id_corpus('Train', self.train_corpus)

        return id_train

    def _to_id_corpus(self, name, data):
        results = []
        for dlg in data:
            if len(dlg.dlg) < 1:
                continue
            id_dlg = []
            for turn in dlg.dlg:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker,
                               db=turn.db, bs=turn.bs)
                id_dlg.append(id_turn)
            id_goal = self._goal2id(dlg.goal)
            results.append(Pack(dlg=id_dlg, goal=id_goal, key=dlg.key))
        return results

    def _sent2id(self, sent):
        return [self.vocab_dict.get(t, self.unk_id) for t in sent]

    def _goal2id(self, goal):
        res = {}
        for domain in self.domains:
            d_bow = [0.0] * len(self.goal_vocab[domain])
            for word in goal[domain]:
                word_id = self.goal_vocab_dict[domain].get(
                    word, self.goal_unk_id[domain])
                d_bow[word_id] += 1.0
            res[domain] = d_bow
        return res

    def id2sent(self, id_list):
        return [self.vocab[i] for i in id_list]

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens


