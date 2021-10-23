#! /usr/bin/env python3

import rospy
import comet_ml
from multiprocessing import set_start_method
import torch.multiprocessing as torch_mp
import multiprocessing as mp
import numpy as np
import queue
import torch
import time
import yaml
import copy
import os
try:
    set_start_method('spawn')
except:
    pass
from utils.utils import empty_torch_queue, create_replay_buffer
from algorithms.d4pg import LearnerD4PG
from tensorboardX import SummaryWriter
from models import PolicyNetwork
from agent import Agent


def sampler_worker(config, replay_queue, batch_queue, replay_priorities_queue, training_on, global_episode, update_step,
                   logs, experiment_dir):
    # Create replay buffer
    replay_buffer = create_replay_buffer(config)
    batch_size = config['batch_size']

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
        with logs.get_lock():
            logs[0] = replay_queue.qsize()
            logs[1] = batch_queue.qsize()
            logs[2] = len(replay_buffer)

    if config['save_buffer']:
        replay_buffer.dump(experiment_dir)

    empty_torch_queue(batch_queue)
    print("Stop sampler worker.")


def logger(config, logs, training_on, update_step, global_episode):
    # Initialize the SummaryWriter
    os.environ['COMET_API_KEY'] = config['api_key']
    comet_ml.init(project_name=config['project_name'])
    writer = SummaryWriter(comet_config={"disabled": True if config['disabled'] else False})
    writer.add_hparams(hparam_dict=config, metric_dict={})
    num_agents = config['num_agents']
    fake_local_eps = np.zeros(num_agents, dtype=np.int)
    fake_data_struct = np.zeros(3)
    fake_step = 0
    while training_on.value:
        step = update_step.value
        if all(fake_data_struct != logs[:3]):
            fake_data_struct[:] = logs[:3]
            writer.add_scalars(main_tag="data_struct", tag_scalar_dict={"global_episode": global_episode.value,
                               "replay_queue": logs[0], "batch_queue": logs[1], "replay_buffer": logs[2]},
                               global_step=step)
        if fake_step != step:
            fake_step = step
            writer.add_scalars(main_tag="losses", tag_scalar_dict={"policy_loss": logs[3], "value_loss": logs[4],
                               "learner_update_timing": logs[5]}, global_step=step)
        for agent in range(num_agents):
            aux = 6 + agent * 3
            if fake_local_eps[agent] != logs[aux + 2]:
                fake_local_eps[agent] = logs[aux + 2]
                writer.add_scalars(main_tag="agent_{}".format(agent), tag_scalar_dict={"reward": logs[aux],
                                   "episode_timing": logs[aux + 1], "episode": logs[aux + 2]}, global_step=step)
        time.sleep(0.05)
    writer.close()


def learner_worker(config, training_on, policy, target_policy_net, learner_w_queue, replay_priority_queue, batch_queue,
                   update_step, logs, experiment_dir):
    learner = LearnerD4PG(config, policy, target_policy_net, learner_w_queue, log_dir=experiment_dir)
    learner.run(training_on, batch_queue, replay_priority_queue, update_step, logs)


def agent_worker(config, policy, learner_w_queue, global_episode, i, agent_type, experiment_dir, training_on,
                 replay_queue, logs, update_step):
    agent = Agent(config=config, policy=policy, global_episode=global_episode, n_agent=i, agent_type=agent_type,
                  log_dir=experiment_dir)
    agent.run(training_on, replay_queue, learner_w_queue, update_step, logs)


if __name__ == "__main__":
    # Loading configs from config.yaml
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/configs/config.yml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Opening gazebo environments
    for i in range(config['num_agents']):
        os.system('gnome-terminal --tab --working-directory=WORK_DIR -- zsh -c "export '
                  'ROS_MASTER_URI=http://localhost:{}; export GAZEBO_MASTER_URI=http://localhost:{}; roslaunch '
                  'turtlebot3_gazebo turtlebot3_stage_{}.launch"'.format(11310 + i, 11340 + i, config['env_stage']))
        time.sleep(2)
    time.sleep(10)

    if config['seed']:
        torch.manual_seed(config['random_seed'])
        np.random.seed(config['random_seed'])

    # Create directory for experiment
    experiment_dir = path + '/saved_models/'
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Data structures
    processes = []
    replay_queue = mp.Queue(maxsize=config['replay_queue_size'])
    training_on = mp.Value('i', 1)
    update_step = mp.Value('i', 0)
    global_episode = mp.Value('i', 0)
    logs = mp.Array('d', np.zeros(6 + 3 * config['num_agents']))
    learner_w_queue = torch_mp.Queue(maxsize=config['num_agents'])
    replay_priorities_queue = mp.Queue(maxsize=config['replay_queue_size'])

    # Logger
    p = torch_mp.Process(target=logger, args=(config, logs, training_on, update_step, global_episode))
    processes.append(p)

    # Data sampler
    batch_queue = mp.Queue(maxsize=config['batch_queue_size'])
    p = torch_mp.Process(target=sampler_worker, args=(config, replay_queue, batch_queue, replay_priorities_queue,
                                                      training_on, global_episode, update_step, logs, experiment_dir))
    processes.append(p)

    # Learner (neural net training process)
    target_policy_net = PolicyNetwork(config['state_dim'], config['action_dim'], config['dense_size'],
                                      device=config['device'])
    policy_net = copy.deepcopy(target_policy_net)
    policy_net_cpu = PolicyNetwork(config['state_dim'], config['action_dim'], config['dense_size'],
                                   device=config['device'])
    target_policy_net.share_memory()

    p = torch_mp.Process(target=learner_worker, args=(config, training_on, policy_net, target_policy_net,
                         learner_w_queue, replay_priorities_queue, batch_queue, update_step, logs, experiment_dir))
    processes.append(p)

    # Single agent for exploitation
    p = torch_mp.Process(target=agent_worker, args=(config, target_policy_net, None, global_episode, 0, "exploitation",
                                                    experiment_dir, training_on, replay_queue, logs, update_step))
    processes.append(p)

    # Agents (exploration processes)
    for i in range(1, config['num_agents']):
        p = torch_mp.Process(target=agent_worker, args=(config, copy.deepcopy(policy_net_cpu), learner_w_queue,
                             global_episode, i, "exploration", experiment_dir, training_on, replay_queue, logs,
                             update_step))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("End.")
