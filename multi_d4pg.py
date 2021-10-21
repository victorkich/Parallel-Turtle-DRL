#! /usr/bin/env python3

import rospy
import comet_ml
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
from tensorboardX import SummaryWriter
from models import PolicyNetwork
from agent import Agent
import numpy as np
import torch
import yaml
import copy
import os


# Helper function to display logged assets in the Comet UI
def display(tab=None):
    experiment = comet_ml.get_global_experiment()
    experiment.display(tab=tab)


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


def sampler_worker(config, replay_queue, batch_queue, replay_priorities_queue, training_on, global_episode, update_step,
                   writer, experiment_dir):

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
        step = update_step.value
        writer.add_scalars(
            "data/data_struct",
            {
                "global_episode": global_episode.value,
                "replay_queue": replay_queue.qsize(),
                "batch_queue": batch_queue.qsize(),
                "replay_buffer": len(replay_buffer)
            },
            step
        )

    if config['save_buffer']:
        replay_buffer.dump(experiment_dir)

    empty_torch_queue(batch_queue)
    print("Stop sampler worker.")


def learner_worker(config, training_on, policy, target_policy_net, learner_w_queue, replay_priority_queue, batch_queue,
                   update_step, writer, experiment_dir):
    learner = LearnerD4PG(config, policy, target_policy_net, learner_w_queue, writer, log_dir=experiment_dir)
    learner.run(training_on, batch_queue, replay_priority_queue, update_step)


def agent_worker(config, policy, learner_w_queue, global_episode, i, agent_type, experiment_dir, training_on,
                 replay_queue, update_step, writer):
    agent = Agent(config=config, policy=policy, global_episode=global_episode, writer=writer, n_agent=i,
                  agent_type=agent_type, log_dir=experiment_dir)
    agent.run(training_on, replay_queue, learner_w_queue, update_step)


if __name__ == "__main__":
    comet_ml.init(project_name='tensorboardX')

    # Loading configs from config.yaml
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/config/config.yml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Initialize the SummaryWriter
    writer = SummaryWriter(comet_config={"disabled": False})

    # Opening gazebo environments
    for i in range(config['num_agents']):
        os.system('gnome-terminal --tab --working-directory=WORK_DIR -- bash -c "export '
                  'ROS_MASTER_URI=http://localhost:{}; export GAZEBO_MASTER_URI=http://localhost:{}; roslaunch '
                  'turtlebot3_gazebo_{}.launch"'.format(11310 + i, 11340 + i, config['env_stage']))
        time.sleep(2)

    # Adjust as much as you need, limited number of physical cores of your cpu
    env_name = 'TurtleBot3_Circuit_Simple-v0'
    script_name = os.path.basename(__file__)

    if config['seed']:
        torch.manual_seed(config['random_seed'])
        np.random.seed(config['random_seed'])

    # Create directory for experiment
    experiment_dir = path + 'saved_models/'
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Data structures
    processes = []
    replay_queue = mp.Queue(maxsize=config['replay_queue_size'])
    training_on = mp.Value('i', 1)
    update_step = mp.Value('i', 0)
    global_episode = mp.Value('i', 0)
    learner_w_queue = torch_mp.Queue(maxsize=config['num_agents'])
    replay_priorities_queue = mp.Queue(maxsize=config['replay_queue_size'])

    # Data sampler
    batch_queue = mp.Queue(maxsize=config['batch_queue_size'])
    p = torch_mp.Process(target=sampler_worker, args=(config, replay_queue, batch_queue, replay_priorities_queue,
                                                      training_on, global_episode, update_step, writer, experiment_dir))
    processes.append(p)

    # Learner (neural net training process)
    target_policy_net = PolicyNetwork(config['state_dim'], config['action_dim'], config['dense_size'],
                                      device=config['device'])
    policy_net = copy.deepcopy(target_policy_net)
    policy_net_cpu = PolicyNetwork(config['state_dim'], config['action_dim'], config['dense_size'],
                                   device=config['device'])
    target_policy_net.share_memory()

    p = torch_mp.Process(target=learner_worker, args=(config, training_on, policy_net, target_policy_net,
                         learner_w_queue, replay_priorities_queue, batch_queue, update_step, writer, experiment_dir))
    processes.append(p)

    # Single agent for exploitation
    p = torch_mp.Process(target=agent_worker, args=(config, target_policy_net, None, global_episode, 0, "exploitation",
                                                    experiment_dir, training_on, replay_queue, update_step, writer))
    processes.append(p)

    # Agents (exploration processes)
    for i in range(1, config['num_agents']):
        p = torch_mp.Process(target=agent_worker, args=(config, copy.deepcopy(policy_net_cpu), learner_w_queue,
                             global_episode, i, "exploration", experiment_dir, training_on, replay_queue, update_step,
                             writer))
        processes.append(p)

    for p in processes:
        p.start()
        time.sleep(0.25)
    for p in processes:
        p.join()

    print("End.")
