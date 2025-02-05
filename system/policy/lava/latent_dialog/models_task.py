import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from system.policy.lava.latent_dialog.corpora import SYS, EOS, PAD, BOS
from system.policy.lava.latent_dialog.utils import INT, FLOAT, LONG, Pack, cast_type
from system.policy.lava.latent_dialog.enc2dec.encoders import RnnUttEncoder
from system.policy.lava.latent_dialog.enc2dec.decoders import DecoderRNN
from system.policy.lava.latent_dialog import nn_lib

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.use_gpu = config.use_gpu
        self.config = config
        self.kl_w = 0.0

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        return cast_type(Variable(th.from_numpy(inputs)), 
                         dtype, 
                         self.use_gpu)

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, loss, batch_cnt):
        total_loss = self.valid_loss(loss, batch_cnt)
        total_loss.backward()

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = 0.0
        for k, l in loss.items():
            if l is not None:
                total_loss += l
        return total_loss

    def get_optimizer(self, config, verbose=True):
        if config.op == 'adam':
            if verbose:
                print('Use Adam')
            return optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=config.init_lr, weight_decay=config.l2_norm)
        elif config.op == 'sgd':
            print('Use SGD')
            return optim.SGD(self.parameters(), lr=config.init_lr, momentum=config.momentum)
        elif config.op == 'rmsprop':
            print('Use RMSProp')
            return optim.RMSprop(self.parameters(), lr=config.init_lr, momentum=config.momentum)

    def get_clf_optimizer(self, config):
        params = []
        params.extend(self.gru_attn_encoder.parameters())
        params.extend(self.feat_projecter.parameters())
        params.extend(self.sel_classifier.parameters())

        if config.fine_tune_op == 'adam':
            print('Use Adam')
            return optim.Adam(params, lr=config.fine_tune_lr)
        elif config.fine_tune_op == 'sgd':
            print('Use SGD')
            return optim.SGD(params, lr=config.fine_tune_lr, momentum=config.fine_tune_momentum)
        elif config.fine_tune_op == 'rmsprop':
            print('Use RMSProp')
            return optim.RMSprop(params, lr=config.fine_tune_lr, momentum=config.fine_tune_momentum)

        
    def model_sel_loss(self, loss, batch_cnt):
        return self.valid_loss(loss, batch_cnt)


    def extract_short_ctx(self, context, context_lens, backward_size=1):
        utts = []
        for b_id in range(context.shape[0]):
            utts.append(context[b_id, context_lens[b_id]-1])
        return np.array(utts)

    def flatten_context(self, context, context_lens, align_right=False):
        utts = []
        temp_lens = []
        for b_id in range(context.shape[0]):
            temp = []
            for t_id in range(context_lens[b_id]):
                for token in context[b_id, t_id]:
                    if token != 0:
                        temp.append(token)
            temp_lens.append(len(temp))
            utts.append(temp)
        max_temp_len = np.max(temp_lens)
        results = np.zeros((context.shape[0], max_temp_len))
        for b_id in range(context.shape[0]):
            if align_right:
                results[b_id, -temp_lens[b_id]:] = utts[b_id]
            else:
                results[b_id, 0:temp_lens[b_id]] = utts[b_id]

        return results

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 


class SysPerfectBD2Cat(BaseModel):
    def __init__(self, corpus, config):
        super(SysPerfectBD2Cat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior
        self.contextual_posterior = config.contextual_posterior

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                          config.y_size, config.k_size, is_lstm=False)
        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)

        if not self.simple_posterior:
            if self.contextual_posterior:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                                   config.y_size, config.k_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

    def forward_rl(self, data_feed, max_words, py_temp=0.1, dec_temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)

        qy = F.softmax(logits_py / py_temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )
        idx = th.multinomial(qy, 1).detach()
        logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
        joint_logpz = th.sum(logprob_sample_z, dim=1)
        sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
        sample_y.scatter_(1, idx, 1.0)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=dec_temp)
        return logprobs, outs, joint_logpz, sample_y

