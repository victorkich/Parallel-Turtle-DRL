#! /usr/bin/env python3

import rospy
from multiprocessing import set_start_method
import torch.multiprocessing as torch_mp
import multiprocessing as mp
import time
import queue
try:
    set_start_method('spawn')
except:
    pass

from utils.utils import empty_torch_queue, create_replay_buffer
from algorithms.d4pg import LearnerD4PG
from models import PolicyNetwork
from agent import Agent
import numpy as np
import argparse
import torch
import copy
import os


def update_actor_learner(self, learner_w_queue, training_on):
    """Update local actor to the actor from learner. """
    if not training_on.value:
        return
    try:
        source = learner_w_queue.get_nowait()
    except:
        return
    target = self.actor
    for target_param, source_param in zip(target.parameters(), source):
        w = torch.tensor(source_param).float()
        target_param.data.copy_(w)
    del source


def sampler_worker(replay_queue, batch_queue, replay_priorities_queue, training_on, global_episode, update_step,
                   log_dir=''):
    batch_size = 256

    # Create replay buffer
    replay_buffer = create_replay_buffer(log_dir)

    while training_on.value:
        # (1) Transfer replays to global buffer
        n = replay_queue.qsize()
        for _ in range(n):
            replay = replay_queue.get()
            replay_buffer.add(*replay)

        # (2) Transfer batch of replay from buffer to the batch_queue
        if len(replay_buffer) < batch_size:
            continue

        try:
            inds, weights = replay_priorities_queue.get_nowait()
            replay_buffer.update_priorities(inds, weights)
        except queue.Empty:
            pass

        try:
            batch = replay_buffer.sample(batch_size)
            batch_queue.put_nowait(batch)
        except:
            time.sleep(0.1)
            continue

        # Log data structures sizes
        step = update_step.value

    if args.save_buffer_on_disk:
        replay_buffer.dump(args.results_path)

    empty_torch_queue(batch_queue)
    print("Stop sampler worker.")


def learner_worker(training_on, policy, target_policy_net, learner_w_queue, replay_priority_queue, batch_queue,
                   update_step, experiment_dir):
    learner = LearnerD4PG(policy, target_policy_net, learner_w_queue, log_dir=experiment_dir)
    learner.run(training_on, batch_queue, replay_priority_queue, update_step)


def save(self, checkpoint_name):
    process_dir = f"{self.log_dir}/agent_{self.n_agent}"
    if not os.path.exists(process_dir):
        os.makedirs(process_dir)
    model_fn = f"{process_dir}/{checkpoint_name}.pt"
    torch.save(self.actor, model_fn)


def agent_worker(policy, learner_w_queue, global_episode, i, agent_type, experiment_dir, training_on, replay_queue,
                 update_step):
    agent = Agent(policy=policy, global_episode=global_episode, n_agent=i, agent_type=agent_type, log_dir=experiment_dir)
    agent.run(training_on, replay_queue, learner_w_queue, update_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
    parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
    parser.add_argument('--target_update_interval', default=1, type=int)
    parser.add_argument('--test_iteration', default=10, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
    parser.add_argument('--capacity', default=1000000, type=int)  # replay buffer size
    parser.add_argument('--batch_size', default=256, type=int)  # mini batch size
    parser.add_argument('--seed', default=False, type=bool)
    parser.add_argument('--random_seed', default=9527, type=int)

    # optional parameters
    parser.add_argument('--sample_frequency', default=2000, type=int)
    parser.add_argument('--log_interval', default=50, type=int)  #
    parser.add_argument('--load', default=False, type=bool)  # load model
    parser.add_argument('--exploration_noise', default=0.25, type=float)
    parser.add_argument('--max_episode', default=100000, type=int)  # num of games
    parser.add_argument('--max_steps', default=500, type=int)
    parser.add_argument('--print_log', default=5, type=int)
    parser.add_argument('--update_iteration', default=200, type=int)
    parser.add_argument('--parallel_environments', default=2, type=int)
    parser.add_argument('--ros_environment', default='turtlebot3_stage_1.launch', type=str)
    parser.add_argument('--save_buffer_on_disk', default=False, type=bool)
    parser.add_argument('--results_path', default='saved_models', type=str)
    args = parser.parse_args()

    # Adjust as much as you need, limited number of physical cores of your cpu
    parallel_environments = args.parallel_environments
    env_name = 'TurtleBot3_Circuit_Simple-v0'

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print('Using:', device)
    script_name = os.path.basename(__file__)

    if args.seed:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    state_dim = 26
    max_action = 1.5
    action_dim = 2
    # action_low = -1.5
    # action_high = 1.5
    n_step_returns = 5  # number of future steps to collect experiences for N-step returns
    discount_rate = 0.99  # Discount rate (gamma) for future rewards
    update_agent_ep = 1  # agent gets latest parameters from learner every update_agent_ep episodes
    dense_size = 128  # size of the 2 hidden layers in networks

    # freeze_support()
    batch_queue_size = 64  # queue with batches given to learner

    # Create directory for experiment
    experiment_dir = 'saved_models/'
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Data structures
    processes = []
    replay_queue = mp.Queue(maxsize=64)
    training_on = mp.Value('i', 1)
    update_step = mp.Value('i', 0)
    global_episode = mp.Value('i', 0)
    learner_w_queue = torch_mp.Queue(maxsize=parallel_environments)
    replay_priorities_queue = mp.Queue(maxsize=64)

    # Data sampler
    batch_queue = mp.Queue(maxsize=batch_queue_size)
    p = torch_mp.Process(target=sampler_worker, args=(replay_queue, batch_queue, replay_priorities_queue, training_on,
                   global_episode, update_step, experiment_dir))
    processes.append(p)

    # Learner (neural net training process)
    target_policy_net = PolicyNetwork(state_dim, action_dim, dense_size, device=device)
    policy_net = copy.deepcopy(target_policy_net)
    policy_net_cpu = PolicyNetwork(state_dim, action_dim, dense_size, device=device)
    target_policy_net.share_memory()

    p = torch_mp.Process(target=learner_worker, args=(training_on, policy_net, target_policy_net, learner_w_queue,
                         replay_priorities_queue, batch_queue, update_step, experiment_dir))
    processes.append(p)

    # Single agent for exploitation
    p = torch_mp.Process(target=agent_worker, args=(target_policy_net, None, global_episode, 0, "exploitation",
                                              experiment_dir, training_on, replay_queue, update_step))
    processes.append(p)

    # Agents (exploration processes)
    for i in range(1, parallel_environments):
        p = torch_mp.Process(target=agent_worker, args=(policy_net_cpu, learner_w_queue, global_episode, i,
                             "exploration", experiment_dir, training_on, replay_queue, update_step))
        processes.append(p)

    for p in processes:
        p.start()
        time.sleep(0.25)
    for p in processes:
        p.join()

    print("End.")
