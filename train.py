#! /usr/bin/env python3

import rospy
import comet_ml
from multiprocessing import set_start_method
from colorama import init as colorama_init
import torch.multiprocessing as torch_mp
import multiprocessing as mp
from colorama import Fore
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
from algorithms.dsac import LearnerDSAC
from algorithms.d4pg import LearnerD4PG
from algorithms.ddpg import LearnerDDPG
from algorithms.sac import LearnerSAC
from tensorboardX import SummaryWriter
from models2 import PolicyNetwork, TanhGaussianPolicy, DiagGaussianActor
from agent import Agent


def sampler_worker(config, replay_queue, batch_queue, replay_priorities_queue, training_on, logs, experiment_dir, update_step):
    torch.set_num_threads(4)
    # Create replay buffer
    replay_buffer = create_replay_buffer(config, experiment_dir)
    batch_size = config['batch_size']

    while training_on.value:
        # (1) Transfer replays to global buffer
        time.sleep(0.3)
        n = replay_queue.qsize()

        for _ in range(n):
            replay = replay_queue.get()
            replay_buffer.add(*replay)

        # (2) Transfer batch of replay from buffer to the batch_queue
        if len(replay_buffer) < batch_size:
            continue

        if len(replay_buffer) > config['replay_mem_size']:
            replay_buffer.remove(len(replay_buffer)-config['replay_mem_size'])

        try:
            if config['replay_memory_prioritized']:
                inds, weights = replay_priorities_queue.get_nowait()
                replay_buffer.update_priorities(inds, weights)
        except queue.Empty:
            pass

        if update_step.value >= config['num_steps_train']:
            beta = config['priority_beta_end']
        else:
            beta = config['priority_beta_start'] + (config['priority_beta_end']-config['priority_beta_start']) * \
                   (update_step.value / config['num_steps_train'])
        try:
            batch = replay_buffer.sample(batch_size, beta=beta)
            batch_queue.put_nowait(batch)
        except queue.Full:
            pass

        try:
            # Log data structures sizes
            with logs.get_lock():
                logs[0] = replay_queue.qsize()
                logs[1] = batch_queue.qsize()
                logs[2] = len(replay_buffer)
        except:
            pass

    if config['save_buffer']:
        process_dir = f"{experiment_dir}/{config['model']}_{config['dense_size']}_A{config['num_agents']}_S{config['env_stage']}_{'P' if config['replay_memory_prioritized'] else 'N'}/"
        replay_buffer.dump(process_dir)

    empty_torch_queue(batch_queue)
    print("Stop sampler worker.")


def logger(config, logs, training_on, update_step, global_episode, global_step, log_dir):
    # Initialize the SummaryWriter
    os.environ['COMET_API_KEY'] = config['api_key']
    comet_ml.init(project_name=config['project_name'])
    writer = SummaryWriter(comet_config={"disabled": True if config['disabled'] else False})
    writer.add_hparams(hparam_dict=config, metric_dict={})
    num_agents = config['num_agents']
    fake_local_eps = np.zeros(num_agents, dtype=int)
    fake_step = 0
    os.system('rosclean purge -y')
    print("Starting log...")
    while training_on.value if not config['test'] else (logs[8] <= config['test_trials']):
        try:
            if not config['test']:
                step = update_step.value
                writer.add_scalars(main_tag="data_struct", tag_scalar_dict={"global_episode": global_episode.value,
                                   "global_step": global_step.value, "replay_queue": logs[0], "batch_queue": logs[1],
                                   "replay_buffer": logs[2]}, global_step=step)
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
            else:
                writer.add_scalars(main_tag="agent_0", tag_scalar_dict={"reward": logs[0], "episode_timing": logs[1],
                                                                        "episode": logs[2], "x": logs[3],
                                                                        "y": logs[4]}, global_step=global_step.value)

            time.sleep(0.1)
            writer.flush()
        except:
            print('Error on Logger!')
            pass

    print("Writer closing...")
    process_dir = f"{log_dir}/{config['model']}_{config['dense_size']}_A{config['num_agents']}_S{config['env_stage']}_" \
                  f"{'P' if config['replay_memory_prioritized'] else 'N'}{'_LSTM' if config['recurrent_policy'] else ''}"
    if not os.path.exists(process_dir):
        os.makedirs(process_dir)
    writer.export_scalars_to_json(f"{process_dir}/writer_data.json")
    writer.close()
    print('Logger closed!')


def learner_worker(config, training_on, policy, target_policy_net, learner_w_queue, replay_priority_queue, batch_queue,
                   update_step, global_episode, logs, experiment_dir):
    if config['model'] == 'PDDRL':
        learner = LearnerD4PG(config, policy, target_policy_net, learner_w_queue, log_dir=experiment_dir)
    elif config['model'] =='PDSRL':
        learner = LearnerDSAC(config, policy, target_policy_net, learner_w_queue, log_dir=experiment_dir)
    elif config['model'] =='DDPG':
        learner = LearnerDDPG(config, policy, target_policy_net, learner_w_queue, log_dir=experiment_dir)
    elif config['model'] =='SAC':
        learner = LearnerSAC(config, policy, target_policy_net, learner_w_queue, log_dir=experiment_dir)
    learner.run(training_on, batch_queue, replay_priority_queue, update_step, global_episode, logs)


def agent_worker(config, policy, learner_w_queue, global_episode, i, agent_type, experiment_dir, training_on,
                 replay_queue, logs, global_step):
    agent = Agent(config=config, policy=policy, global_episode=global_episode, n_agent=i, agent_type=agent_type,
                  log_dir=experiment_dir, global_step=global_step)
    agent.run(training_on=training_on, replay_queue=replay_queue, learner_w_queue=learner_w_queue, logs=logs)


if __name__ == "__main__":
    os.system('clear')
    colorama_init(autoreset=True)
    print(Fore.RED + '------ PARALLEL DEEP REINFORCEMENT LEARNING USING PYTORCH ------'.center(100))

    # Loading configs from config.yaml
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/config.yml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    if config['seed']:
        torch.manual_seed(config['random_seed'])
        np.random.seed(config['random_seed'])

    # Create directory for experiment
    experiment_dir = path + '/saved_models/'
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    results_dir = path + f"/{config['results_path']}/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model_name = f"{config['model']}_{config['dense_size']}_A{config['num_agents']}_S{config['env_stage']}_{'P' if config['replay_memory_prioritized'] else 'N'}"
    model_dir = f"{experiment_dir}/{model_name}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    list_saved_models = os.listdir(model_dir)
    higher = 0
    higher_model = None
    for saved_model in list_saved_models:
        if saved_model == 'writer_data.json':
            continue
        if higher < int(saved_model.split('_')[1].split('.')[0]):
            higher = int(saved_model.split('_')[1].split('.')[0])
            higher_model = saved_model
    path_model = f"{model_dir}{higher_model}"
    print(path_model)
    # path_model = f"{model_dir}ep"

    # Opening gazebo environments
    for i in range(config['num_agents'] if not config['test'] else 1):
        if not i:
            os.system('gnome-terminal --tab --working-directory=WORK_DIR -- zsh -c "export '
                      'ROS_MASTER_URI=http://localhost:{}; export GAZEBO_MASTER_URI=http://localhost:{}; roslaunch '
                      'turtlebot3_gazebo turtlebot3_stage_{}_1.launch"'.format(11311 + i, 11341 + i,
                                                                               config['env_stage']))
        else:
            os.system('gnome-terminal --tab --working-directory=WORK_DIR -- zsh -c "export '
                      'ROS_MASTER_URI=http://localhost:{}; export GAZEBO_MASTER_URI=http://localhost:{}; roslaunch '
                      'turtlebot3_gazebo turtlebot3_stage_{}.launch"'.format(11311 + i, 11341 + i, config['env_stage']))
        time.sleep(1)
    time.sleep(5)

    # Data structures
    processes = []
    replay_queue = mp.Queue(maxsize=config['replay_queue_size'])
    training_on = mp.Value('i', 1)
    update_step = mp.Value('i', 0)
    global_episode = mp.Value('i', 0)
    global_step = mp.Value('i', 0)
    logs = mp.Array('d', np.zeros(6 + 3 * config['num_agents']))
    learner_w_queue = torch_mp.Queue(maxsize=config['num_agents'])
    replay_priorities_queue = mp.Queue(maxsize=config['replay_queue_size'])

    # Logger
    p = torch_mp.Process(target=logger, args=(config, logs, training_on, update_step, global_episode, global_step,
                                              experiment_dir if not config['test'] else results_dir))
    processes.append(p)

    # Data sampler
    if not config['test']:
        batch_queue = mp.Queue(maxsize=config['batch_queue_size'])
        p = torch_mp.Process(target=sampler_worker, args=(config, replay_queue, batch_queue, replay_priorities_queue,
                                                          training_on, logs, experiment_dir, update_step))
        processes.append(p)

    # Learner (neural net training process)
    if config['model'] == 'PDDRL' or config['model'] == 'DDPG':
        if config['test']:
            try:
                target_policy_net = PolicyNetwork(config['state_dim'], config['action_dim'], config['dense_size'],
                                                  device=config['device'], recurrent=config['recurrent_policy'],
                                                  lstm_cells=config['num_lstm_cell'])
                target_policy_net.load_state_dict(torch.load(path_model, map_location=config['device']))
            except:
                target_policy_net = torch.load(path_model)
                target_policy_net.to(config['device'])
            target_policy_net.eval()
        else:
            target_policy_net = PolicyNetwork(config['state_dim'], config['action_dim'], config['dense_size'],
                                              device=config['device'], recurrent=config['recurrent_policy'],
                                              lstm_cells=config['num_lstm_cell'])
            policy_net = copy.deepcopy(target_policy_net)
            policy_net_cpu = PolicyNetwork(config['state_dim'], config['action_dim'], config['dense_size'], device=config['device'],
                                           recurrent=config['recurrent_policy'], lstm_cells=config['num_lstm_cell'])
    elif config['model'] == 'PDSRL':
        if config['test']:
            try:
                target_policy_net = TanhGaussianPolicy(config=config, obs_dim=config['state_dim'], action_dim=config['action_dim'],
                                                       hidden_sizes=[config['dense_size'], config['dense_size']],
                                                       recurrent=config['recurrent_policy'], lstm_cells=config['num_lstm_cell'])
                target_policy_net.load_state_dict(torch.load(path_model, map_location=config['device']))
            except:
                target_policy_net = torch.load(path_model)
                target_policy_net.to(config['device'])
            target_policy_net.eval()
        else:
            target_policy_net = TanhGaussianPolicy(config=config, obs_dim=config['state_dim'], action_dim=config['action_dim'],
                                                   hidden_sizes=[config['dense_size'], config['dense_size']],
                                                   recurrent=config['recurrent_policy'], lstm_cells=config['num_lstm_cell'])
            policy_net = copy.deepcopy(target_policy_net)
            policy_net_cpu = TanhGaussianPolicy(config=config, obs_dim=config['state_dim'], action_dim=config['action_dim'],
                                                hidden_sizes=[config['dense_size'], config['dense_size']],
                                                recurrent=config['recurrent_policy'], lstm_cells=config['num_lstm_cell'])
    elif config['model'] == 'SAC':
        if config['test']:
            try:
                target_policy_net = DiagGaussianActor(config['state_dim'], config['action_dim'], config['dense_size'], 2, [-config['max_action'], config['max_action']])
                target_policy_net.load_state_dict(torch.load(path_model, map_location=config['device']))
            except:
                target_policy_net = torch.load(path_model)
                target_policy_net.to(config['device'])
            target_policy_net.eval()
        else:
            target_policy_net = DiagGaussianActor(config['state_dim'], config['action_dim'], config['dense_size'], 2, [-config['max_action'], config['max_action']])
            policy_net = copy.deepcopy(target_policy_net)
            policy_net_cpu = DiagGaussianActor(config['state_dim'], config['action_dim'], config['dense_size'], 2, [-config['max_action'], config['max_action']])

    print(f"Algorithm: {config['model']}-{'P' if config['replay_memory_prioritized'] else 'N'}-{'LSTM' if config['recurrent_policy'] else ''}")

    if not config['test']:
        policy_net.share_memory()
        p = torch_mp.Process(target=learner_worker, args=(config, training_on, policy_net, target_policy_net,
                                                          learner_w_queue, replay_priorities_queue, batch_queue,
                                                          update_step, global_episode, logs, experiment_dir))
        processes.append(p)

    # Single agent for exploitation
    p = torch_mp.Process(target=agent_worker, args=(config, target_policy_net, learner_w_queue,
                         global_episode, 0, "exploitation", experiment_dir, training_on, replay_queue, logs, global_step))
    processes.append(p)

    # Agents (exploration processes)
    if not config['test']:
        for i in range(1, config['num_agents']):
            p = torch_mp.Process(target=agent_worker, args=(config, copy.deepcopy(policy_net_cpu), learner_w_queue,
                                                            global_episode, i, "exploration", experiment_dir,
                                                            training_on, replay_queue, logs, global_step))
            processes.append(p)

    for p in processes:
        p.daemon = True
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("End.")
