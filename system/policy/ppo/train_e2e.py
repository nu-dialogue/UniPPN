# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:14:07 2019
@author: truthless
"""
import os
import sys
import json
import logging
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

import torch
from torch import optim
from torch import multiprocessing as mp
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dialog_agent.env import Environment
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.policy.ppo import PPO
from convlab2.policy.rlmodule import Memory
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.policy.rlmodule import MultiDiscretePolicy, Value
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab2.util.analysis_tool.analyzer import Analyzer

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
sys.path.append(root_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    mp = mp.get_context('spawn')
except RuntimeError:
    pass

class MyPPO(PPO):
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        
        self.save_dir = cfg['save_dir']
        self.save_per_epoch = cfg['save_per_epoch']
        self.update_round = cfg['update_round']
        self.optim_batchsz = cfg['batchsz']
        self.gamma = cfg['gamma']
        self.epsilon = cfg['epsilon']
        self.tau = cfg['tau']

        voc_file = os.path.join(root_dir, 'ConvLab-2/data/multiwoz/sys_da_voc.txt')
        voc_opp_file = os.path.join(root_dir, 'ConvLab-2/data/multiwoz/usr_da_voc.txt')
        self.vector = MultiWozVector(voc_file, voc_opp_file)
        self.policy = MultiDiscretePolicy(self.vector.state_dim, cfg['h_dim'], self.vector.da_dim).to(device=DEVICE)
        self.value = Value(self.vector.state_dim, cfg['hv_dim']).to(device=DEVICE)

        self.policy_optim = optim.RMSprop(self.policy.parameters(), lr=cfg['policy_lr'])
        self.value_optim = optim.Adam(self.value.parameters(), lr=cfg['value_lr'])

def sampler(pid, queue, evt, env, policy, batchsz):
    """
    This is a sampler function, and it will be called by multiprocess.Process to sample data from environment by multiple
    processes.
    :param pid: process id
    :param queue: multiprocessing.Queue, to collect sampled data
    :param evt: multiprocessing.Event, to keep the process alive
    :param env: environment instance
    :param policy: policy network, to generate action from current policy
    :param batchsz: total sampled items
    :return:
    """
    buff = Memory()

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 20
    real_traj_len = 0

    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()

        for t in range(traj_len):

            # [s_dim] => [a_dim]
            s_vec = torch.Tensor(policy.vector.state_vectorize(s))
            a = policy.predict(s)

            # interact with env
            next_s, r, done = env.step(a)

            # a flag indicates ending or not
            mask = 0 if done else 1

            # get reward compared to demostrations
            next_s_vec = torch.Tensor(policy.vector.state_vectorize(next_s))

            # save to queue
            buff.push(s_vec.numpy(), policy.vector.action_vectorize(a), r, next_s_vec.numpy(), mask)

            # update per step
            s = next_s
            real_traj_len = t

            if done:
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length

    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff])
    evt.wait()


def sample(env, policy, batchsz, process_num):
    """
    Given batchsz number of task, the batchsz will be splited equally to each processes
    and when processes return, it merge all data and return
	:param env:
	:param policy:
    :param batchsz:
	:param process_num:
    :return: batch
    """

    # batchsz will be splitted into each process,
    # final batchsz maybe larger than batchsz parameters
    process_batchsz = np.ceil(batchsz / process_num).astype(np.int32)
    # buffer to save all data
    queue = mp.Queue()

    # start processes for pid in range(1, processnum)
    # if processnum = 1, this part will be ignored.
    # when save tensor in Queue, the process should keep alive till Queue.get(),
    # please refer to : https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
    # however still some problem on CUDA tensors on multiprocessing queue,
    # please refer to : https://discuss.pytorch.org/t/cuda-tensors-on-multiprocessing-queue/28626
    # so just transform tensors into numpy, then put them into queue.
    evt = mp.Event()
    processes = []
    for i in range(process_num):
        process_args = (i, queue, evt, env, policy, process_batchsz)
        processes.append(mp.Process(target=sampler, args=process_args))
    for p in processes:
        # set the process as daemon, and it will be killed once the main process is stoped.
        p.daemon = True
        p.start()

    # we need to get the first Memory object and then merge others Memory use its append function.
    pid0, buff0 = queue.get()
    for _ in range(1, process_num):
        pid, buff_ = queue.get()
        buff0.append(buff_)  # merge current Memory into buff0
    evt.set()

    # now buff saves all the sampled data
    buff = buff0

    return buff.get_batch()


def update(env, policy, batchsz, epoch, process_num):
    # sample data asynchronously
    batch = sample(env, policy, batchsz, process_num)

    # data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
    # batch.action: ([1, a_dim], [1, a_dim]...)
    # batch.reward/ batch.mask: ([1], [1]...)
    s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
    a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
    r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
    mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
    batchsz_real = s.size(0)

    policy.update(epoch, batchsz_real, s, a, r, mask)
    print(r.float().mean())

def test(epoch, user_agent, sys_nlu, sys_dst, sys_policy, sys_nlg):
    test_name = f'test.{epoch}'
    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')
    sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')
    analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name=test_name, total_dialog=100)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="config file name")
    parser.add_argument("--load_path", type=str, default="", help="path of model to load")
    parser.add_argument("--batchsz", type=int, default=1024, help="batch size of trajactory sampling")
    parser.add_argument("--epoch", type=int, default=200, help="number of epochs to train")
    parser.add_argument("--process_num", type=int, default=8, help="number of processes of trajactory sampling")
    parser.add_argument("--epochs_per_test", type=int, default=10, help="number of epochs per test")
    args = parser.parse_args()

    # e2e sys
    sys_nlu = BERTNLU(
        mode='usr',
        config_file='multiwoz_usr_context.json',
        model_file="https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/bert_multiwoz_usr_context.zip",
        device=DEVICE
    )
    sys_dst = RuleDST()
    sys_policy = MyPPO(args.config_path)
    sys_policy.load(args.load_path)
    sys_nlg = TemplateNLG(is_user=False, mode='manual')

    # e2e user
    nlu_usr = BERTNLU(
        mode='sys',
        config_file='multiwoz_sys_context.json',
        model_file="https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/bert_multiwoz_sys_context.zip",
        device=DEVICE
    )
    dst_usr = None
    policy_usr = RulePolicy(character='usr')
    nlg_usr = TemplateNLG(is_user=True)
    simulator = PipelineAgent(nlu_usr, dst_usr, policy_usr, nlg_usr, 'user')

    evaluator = MultiWozEvaluator()
    env = Environment(sys_nlu=sys_nlu, sys_dst=sys_dst, sys_nlg=sys_nlg,
                      usr=simulator, evaluator=evaluator)

    for i in tqdm(range(args.epoch)):
        update(env, sys_policy, args.batchsz, i, args.process_num)
        if i % args.epochs_per_test == 0:
            test(epoch=i, user_agent=simulator,
                 sys_nlu=sys_nlu, sys_dst=sys_dst, sys_policy=sys_policy, sys_nlg=sys_nlg)