import torch as th

from nltk import RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

INT = 0
LONG = 1
FLOAT = 2

class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack

    @staticmethod
    def msg_from_dict(dictionary, tokenize, speaker2id, bos_id, eos_id, include_domain=False):
        pack = Pack()
        for k, v in dictionary.items():
            pack[k] = v
        pack['speaker'] = speaker2id[pack.speaker]
        pack['conf'] = dictionary.get('conf', 1.0)
        utt = pack['utt']
        if 'QUERY' in utt or "RET" in utt:
            utt = str(utt)
            # utt = utt.translate(None, ''.join([':', '"', "{", "}", "]", "["]))
            utt = utt.translate(str.maketrans('', '', ''.join([':', '"', "{", "}", "]", "["])))
            utt = str(utt)
        if include_domain:
            pack['utt'] = [bos_id, pack['speaker'], pack['domain']] + tokenize(utt) + [eos_id]
        else:
            pack['utt'] = [bos_id, pack['speaker']] + tokenize(utt) + [eos_id]
        return pack

def get_tokenize():
    return RegexpTokenizer(r'\w+|#\w+|<\w+>|%\w+|[^\w\s]+').tokenize

def get_detokenize():
    return lambda x: TreebankWordDetokenizer().detokenize(x)

def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(th.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(th.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(th.cuda.FloatTensor)
        else:
            raise ValueError('Unknown dtype')
    else:
        if dtype == INT:
            var = var.type(th.IntTensor)
        elif dtype == LONG:
            var = var.type(th.LongTensor)
        elif dtype == FLOAT:
            var = var.type(th.FloatTensor)
        else:
            raise ValueError('Unknown dtype')
    return var
