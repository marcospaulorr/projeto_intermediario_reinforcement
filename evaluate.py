#!/usr/bin/env python3
"""
Script para avaliar SOMENTE o PPO no ambiente DSSE Coverage
"""

import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from DSSE import CoverageDroneSwarmSearch
from agent import PPOAgent
from utils import create_directories, plot_learning_curves

def evaluate_ppo(agents, drones_positions, num_episodes=10, render=False):
    """Avalia agentes PPO treinados e coleta métricas detalhadas"""
    render_mode = "human" if render else "ansi"
    env = CoverageDroneSwarmSearch(
        drone_amount=len(agents),
        render_mode=render_mode,
        disaster_position=(-24.04, -46.17),
        pre_render_time=2
    )
    
    episodic_metrics = {
        'reward': [],
        'steps': [],
        'coverage_rate': [],
        'cumulative_pos': [],
        'repeated_coverage': []
    }
    detailed_metrics = {
        'reward': [],
        'coverage_rate': [],
        'cumulative_pos': [],
        'repeated_coverage': []
    }
    
    for _ in tqdm(range(num_episodes), desc="Avaliando PPO"):
        options = {"drones_positions": drones_positions}
        observations, _ = env.reset(options=options)
        
        # formatar obs
        obs_dict = {
            drone_id: (np.array(pos, dtype=np.float32), pmatrix)
            for drone_id, (pos, pmatrix) in observations.items()
        }
        
        ep_reward = 0
        ep_steps = 0
        done = False
        last_cov = last_rep = last_cum = 0
        
        while not done:
            # escolher ações
            actions = {
                drone_id: agent.act(obs_dict[drone_id], training=False)
                for drone_id, agent in agents.items()
                if drone_id in env.agents
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # atualizar obs
            for drone_id, (pos, pmatrix) in observations.items():
                obs_dict[drone_id] = (np.array(pos, dtype=np.float32), pmatrix)
            
            step_reward = sum(rewards.values())
            ep_reward += step_reward
            ep_steps += 1
            
            # detalhado por passo
            detailed_metrics['reward'].append(step_reward)
            for info in infos.values():
                if 'coverage_rate' in info:
                    last_cov = info['coverage_rate']
                    detailed_metrics['coverage_rate'].append(last_cov)
                if 'repeated_coverage' in info:
                    last_rep = info['repeated_coverage']
                    detailed_metrics['repeated_coverage'].append(last_rep)
                if 'accumulated_pos' in info:
                    last_cum = info['accumulated_pos']
                    detailed_metrics['cumulative_pos'].append(last_cum)
                break
            
            done = any(terminations.values()) or any(truncations.values()) or not env.agents
        
        episodic_metrics['reward'].append(ep_reward)
        episodic_metrics['steps'].append(ep_steps)
        episodic_metrics['coverage_rate'].append(last_cov)
        episodic_metrics['repeated_coverage'].append(last_rep)
        episodic_metrics['cumulative_pos'].append(last_cum)
    
    env.close()
    return episodic_metrics, detailed_metrics


def compare_algorithms(agents_dir, num_drones=2, num_episodes=100,
                       render=False, results_dir="./results"):
    """Carrega somente PPO e gera curvas de aprendizado"""
    create_directories([results_dir])
    
    # pegar posições iniciais
    temp_env = CoverageDroneSwarmSearch(
        drone_amount=num_drones,
        render_mode="ansi",
        disaster_position=(-24.04, -46.17),
        pre_render_time=2
    )
    grid = temp_env.grid_size
    prob = temp_env.probability_matrix.get_matrix()
    temp_env.close()
    valid = [(x, y) for y in range(grid) for x in range(grid) if prob[y, x] > 0]
    drones_pos = (valid[:num_drones]
                  if len(valid) >= num_drones
                  else valid * ((num_drones + len(valid) - 1) // len(valid)))
    
    # carregar agentes PPO
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agents = {}
    for i in range(num_drones):
        aid = f"drone{i}"
        latest = None
        best_ep = -1
        for f in os.listdir(agents_dir):
            if f.startswith(aid) and f.endswith(".pt"):
                try:
                    ep = int(f.split("ep")[-1].split(".")[0])
                except:
                    continue
                if ep > best_ep:
                    best_ep = ep
                    latest = f
        if latest:
            agents[aid] = PPOAgent.create_from_checkpoint(
                os.path.join(agents_dir, latest), device=device
            )
        else:
            agents[aid] = PPOAgent(
                obs_shape=(grid, grid),
                action_size=8,
                device=device
            )
    
    # roda só o PPO
    episodic, detailed = evaluate_ppo(agents, drones_pos, num_episodes, render)
    
    # **Aqui** embrulhamos em um nível extra para satisfazer plot_learning_curves
    episodic_results = {'PPO': episodic}
    detailed_results = {'PPO': detailed}
    
    # Gerar curvas de aprendizado com a nova função
    for metric in ['reward', 'coverage_rate', 'repeated_coverage', 'cumulative_pos']:
        plot_learning_curves(
            detailed_results,
            metric,
            os.path.join(results_dir, f"learning_curve_{metric}.png")
        )
    
    # Salvar CSV
    rows = []
    for metric_name, vals in episodic.items():
        for i, v in enumerate(vals):
            rows.append({'algorithm': 'PPO',
                         'episode': i,
                         'metric': metric_name,
                         'value': v})
    for metric_name, vals in detailed.items():
        for i, v in enumerate(vals):
            rows.append({'algorithm': 'PPO',
                         'step': i,
                         'metric': metric_name,
                         'value': v})
    pd.DataFrame(rows).to_csv(
        os.path.join(results_dir, "detailed_results.csv"),
        index=False
    )
    
    return episodic_results, detailed_results
