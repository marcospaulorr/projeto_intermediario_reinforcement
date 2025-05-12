#!/usr/bin/env python3
"""
Script para avaliar o modelo PPO treinado no ambiente DSSE Coverage
Compatível com modelos do stable-baselines3
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from stable_baselines3 import PPO
from train import DSSECoverageEnv
from utils import create_directories, plot_learning_curves

def evaluate_ppo_model(model_path, num_drones=2, num_episodes=10, render=False):
    """Avalia um modelo PPO treinado com stable-baselines3"""
    # Criar ambiente de avaliação
    env_kwargs = {
        "disaster_position": (-24.04, -46.17),
        "pre_render_time": 2
    }
    
    render_mode = "human" if render else "ansi"
    env = DSSECoverageEnv(
        drone_amount=num_drones,
        render_mode=render_mode,
        **env_kwargs
    )
    
    # Carregar modelo treinado
    model = PPO.load(model_path)
    print(f"Modelo carregado: {model_path}")
    
    # Métricas para avaliação
    metrics = {
        'reward': [],
        'steps': [],
        'coverage_rate': [],
        'cumulative_pos': [],
        'repeated_coverage': []
    }
    
    # Dados detalhados por passo
    detailed = {
        'reward': [],
        'coverage_rate': [],
        'cumulative_pos': [],
        'repeated_coverage': []
    }
    
    # Executar avaliação
    for ep in tqdm(range(num_episodes), desc="Avaliando modelo"):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        
        # Valores para acompanhamento
        last_cov = 0
        last_rep = 0
        last_cum = 0
        
        while not done:
            # Predizer ação determinística
            action, _ = model.predict(obs, deterministic=True)
            
            # Executar passo
            obs, reward, done, info = env.step(action)
            
            # Registrar recompensa
            ep_reward += reward
            ep_steps += 1
            
            # Registrar métricas detalhadas
            detailed['reward'].append(reward)
            
            # Extrair métricas de cobertura dos drones
            if 'drones_info' in info:
                for drone_info in info['drones_info'].values():
                    if 'coverage_rate' in drone_info:
                        last_cov = drone_info['coverage_rate']
                        detailed['coverage_rate'].append(last_cov)
                    if 'repeated_coverage' in drone_info:
                        last_rep = drone_info['repeated_coverage']
                        detailed['repeated_coverage'].append(last_rep)
                    if 'accumulated_pos' in drone_info:
                        last_cum = drone_info['accumulated_pos']
                        detailed['cumulative_pos'].append(last_cum)
                    break  # Apenas um drone é suficiente para as métricas
        
        # Registrar métricas do episódio
        metrics['reward'].append(ep_reward)
        metrics['steps'].append(ep_steps)
        metrics['coverage_rate'].append(last_cov)
        metrics['repeated_coverage'].append(last_rep)
        metrics['cumulative_pos'].append(last_cum)
        
        if render:
            print(f"Episódio {ep+1}: Reward={ep_reward:.2f}, Steps={ep_steps}, Coverage={last_cov:.2f}%")
    
    # Fechar ambiente
    env.close()
    
    # Resultado em formato compatível com plot_learning_curves
    results = {'PPO': detailed}
    episodic_results = {'PPO': metrics}
    
    return episodic_results, results

def compare_algorithms(agents_dir, num_drones=2, num_episodes=100, render=False, results_dir="./results"):
    """Carrega e avalia o modelo PPO treinado"""
    create_directories([results_dir])
    
    # Encontrar o modelo mais recente
    model_files = [f for f in os.listdir(agents_dir) if f.endswith('.zip')]
    if not model_files:
        print(f"Nenhum modelo encontrado em {agents_dir}")
        return None, None
    
    # Usar o modelo mais recente (com timestamp maior)
    model_path = os.path.join(agents_dir, sorted(model_files)[-1])
    print(f"Usando o modelo: {model_path}")
    
    # Avaliar modelo
    episodic_results, detailed_results = evaluate_ppo_model(
        model_path,
        num_drones=num_drones,
        num_episodes=num_episodes,
        render=render
    )
    
    # Gerar curvas de aprendizado
    for metric in ['reward', 'coverage_rate', 'repeated_coverage', 'cumulative_pos']:
        if metric in detailed_results['PPO']:
            plot_learning_curves(
                detailed_results,
                metric,
                os.path.join(results_dir, f"learning_curve_{metric}.png")
            )
    
    # Salvar resultados como CSV
    rows = []
    for metric_name, vals in episodic_results['PPO'].items():
        for i, v in enumerate(vals):
            rows.append({
                'algorithm': 'PPO',
                'episode': i,
                'metric': metric_name,
                'value': v
            })
    
    for metric_name, vals in detailed_results['PPO'].items():
        for i, v in enumerate(vals):
            rows.append({
                'algorithm': 'PPO',
                'step': i,
                'metric': metric_name,
                'value': v
            })
    
    pd.DataFrame(rows).to_csv(
        os.path.join(results_dir, "detailed_results.csv"),
        index=False
    )
    
    print(f"Avaliação concluída. Resultados salvos em {results_dir}")
    return episodic_results, detailed_results

if __name__ == "__main__":
    # Testar diretamente
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="./models")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--num_drones", type=int, default=4)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    
    args = parser.parse_args()
    
    compare_algorithms(
        args.models_dir,
        num_drones=args.num_drones,
        num_episodes=args.num_episodes,
        render=args.render,
        results_dir=args.results_dir
    )