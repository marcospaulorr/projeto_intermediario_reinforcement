#!/usr/bin/env python3
"""
Script para treinar os agentes PPO no ambiente DSSE Coverage usando stable-baselines3
Adaptado do código do exemplo bem-sucedido
"""

import os
import numpy as np
import torch
import gym
from gym import spaces
from DSSE import CoverageDroneSwarmSearch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from utils import create_directories
from recorder import PygameRecord

# Frequência de print em timesteps
PRINT_FREQ = 10000

class ProgressCallback(BaseCallback):
    """
    Callback para imprimir o número de timesteps a cada PRINT_FREQ interações.
    """
    def __init__(self, print_freq, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print(f"[TREINO] Timestep atual: {self.num_timesteps}")
        return True

class DSSECoverageEnv(gym.Env):
    """
    Adaptação do ambiente DSSE Coverage para interface Gym padrão
    Compatível com biblioteca stable-baselines3
    """
    def __init__(self, drone_amount=2, render_mode="ansi", **env_kwargs):
        super().__init__()
        # Instanciar ambiente DSSE Coverage
        self.env = CoverageDroneSwarmSearch(
            drone_amount=drone_amount, 
            render_mode=render_mode, 
            **env_kwargs, 
            timestep_limit=200
        )
        self.num_drones = drone_amount

        # Reset inicial para obter dimensões do ambiente
        obs_dict, _ = self.env.reset()
        first_agent = sorted(obs_dict.keys())[0]
        _, prob = obs_dict[first_agent]
        H, W = prob.shape
        self.H, self.W = H, W

        # Definir posições iniciais estratégicas (cantos do ambiente)
        self.initial_positions = [
            (0, 0),
            (W - 1, 0),
            (0, H - 1),
            (W - 1, H - 1)
        ][:self.num_drones]  # Limitar ao número de drones
        
        # Se precisar de mais posições, usar células com maior probabilidade
        if self.num_drones > 4:
            prob_matrix = self.env.probability_matrix.get_matrix()
            valid_cells = [(x, y) for y in range(H) for x in range(W) if prob_matrix[y, x] > 0]
            valid_cells.sort(key=lambda pos: prob_matrix[pos[1], pos[0]], reverse=True)
            self.initial_positions.extend(valid_cells[:self.num_drones-4])

        # Reset com posições iniciais
        obs_dict, _ = self.env.reset(options={"drones_positions": self.initial_positions})

        # Combinar observações em um único vetor
        combined_obs = self._combine(obs_dict)
        obs_len = combined_obs.shape[0]
        
        # Definir espaços de observação e ação
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_len,), 
            dtype=np.float32
        )
        
        act_n = self.env.action_space(first_agent).n
        self.action_space = spaces.MultiDiscrete([act_n] * self.num_drones)
        
        # Rastreamento interno
        self.current_obs = combined_obs
        self.visited = np.zeros((H, W), dtype=bool)  # Rastrear células visitadas

    def _combine(self, obs_dict):
        """
        Combina observações de todos os drones em um único vetor
        Formato: [posições de todos drones, matriz de probabilidade achatada]
        """
        agents = sorted(obs_dict.keys())
        pos_list = []
        for agent in agents:
            (x, y), _ = obs_dict[agent]
            pos_list.extend([float(x), float(y)])
        
        # Pegar matriz de probabilidade do primeiro drone (igual para todos)
        _, prob0 = obs_dict[agents[0]]
        flat = prob0.flatten().astype(np.float32)
        
        return np.concatenate([np.array(pos_list, dtype=np.float32), flat])

    def reset(self):
        """Reset do ambiente com posições iniciais definidas"""
        obs_dict, _ = self.env.reset(options={"drones_positions": self.initial_positions})
        self.current_obs = self._combine(obs_dict)
        self.visited = np.zeros((self.H, self.W), dtype=bool)  # Limpar mapa de visitas
        return self.current_obs

    def step(self, action):
        """Executa um passo no ambiente com sistema de recompensa aprimorado"""
        # Converter ação para formato do ambiente original
        if isinstance(action, (list, tuple, np.ndarray)):
            act_dict = {f"drone{i}": int(action[i]) for i in range(self.num_drones)}
        else:
            act_dict = {"drone0": int(action)}
        
        # Executar ação no ambiente
        obs_dict, rewards, term, trunc, infos = self.env.step(act_dict)
        
        # Calcular penalidade por revisitar células
        penalty = 0
        for agent in obs_dict:
            (x, y), _ = obs_dict[agent]
            if self.visited[y, x]:
                penalty -= 0.2  # Penalidade por revisita
            self.visited[y, x] = True  # Marcar como visitada
        
        # Recompensa total = recompensa original + penalidade
        total_reward = sum(rewards.values()) + penalty
        
        # Verificar término
        done = not self.env.agents or all(term.get(a, False) or trunc.get(a, False) for a in obs_dict)
        
        # Informações adicionais
        info = {"drones_info": infos, "penalty": penalty}
        
        # Atualizar observação atual
        self.current_obs = None if done else self._combine(obs_dict)
        
        return self.current_obs, total_reward, done, info

    def render(self, mode="human"):
        """Renderiza o ambiente"""
        try:
            return self.env.render()
        except:
            return None

    def close(self):
        """Fecha o ambiente"""
        try:
            self.env.close()
        except:
            pass

def train(args, experiment_config=None):
    """
    Treina um agente PPO no ambiente DSSE Coverage usando stable-baselines3
    Baseado na implementação bem-sucedida
    """
    print(f"Iniciando treinamento com {args.num_drones} drones")
    
    # Criar diretórios para modelos e logs
    models_dir = args.models_dir
    logs_dir = os.path.join(models_dir, "logs")
    create_directories([models_dir, logs_dir])
    
    # Configurar hiperparâmetros
    if experiment_config:
        lr = experiment_config["learning_rate"]
        gamma = experiment_config["gamma"]
        clip_range = experiment_config["clip_ratio"]
        ent_coef = experiment_config["entropy_coef"]
    else:
        lr = args.learning_rate
        gamma = args.gamma
        clip_range = 0.2
        ent_coef = 0.01
    
    # Criar ambiente adaptado
    env_kwargs = {
        "disaster_position": (-24.04, -46.17),
        "pre_render_time": args.pre_render_time
    }
    
    # Adicionar matriz pré-computada se disponível
    if hasattr(args, "prob_matrix_path") and args.prob_matrix_path:
        env_kwargs["prob_matrix_path"] = args.prob_matrix_path
    
    env = DSSECoverageEnv(
        drone_amount=args.num_drones,
        render_mode="ansi",
        **env_kwargs
    )
    
    # Criar modelo PPO com hyperparameters otimizados
    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        tensorboard_log=logs_dir,
        learning_rate=lr,
        n_steps=1024,             # Tamanho do rollout
        batch_size=args.batch_size,
        n_epochs=args.ppo_epochs,
        gamma=gamma,              # Fator de desconto
        gae_lambda=0.92,          # Lambda para vantagem generalizada
        clip_range=clip_range,    # Clipping ratio
        ent_coef=ent_coef,        # Coeficiente de entropia
        vf_coef=0.5,              # Coeficiente para perda de valor
        max_grad_norm=0.5,        # Clipping de gradiente
    )
    
    # Criar callback para monitoramento
    callback = ProgressCallback(args.plot_freq)
    
    # Treinar o modelo
    print(f"Iniciando treinamento por {args.num_episodes} timesteps...")
    model.learn(
        total_timesteps=args.num_episodes,
        callback=callback
    )
    
    # Salvar modelo
    model_path = os.path.join(models_dir, f"ppo_coverage_{args.num_drones}drones")
    model.save(model_path)
    print(f"Modelo salvo em {model_path}")
    
    # Fechar ambiente
    env.close()
    
    # Criar GIF de demonstração (opcional)
    if args.create_gif:
        create_demonstration_gif(model, args.num_drones, models_dir)
    
    print("Treinamento concluído!")
    return model

def create_demonstration_gif(model, num_drones, save_dir):
    """Cria um GIF demonstrando o comportamento do agente treinado"""
    print("Criando GIF de demonstração...")
    
    # Criar ambiente para avaliação visual
    env = DSSECoverageEnv(
        drone_amount=num_drones,
        render_mode="human",
        disaster_position=(-24.04, -46.17),
        pre_render_time=2
    )
    
    # Preparar gravação
    gif_path = os.path.join(save_dir, f"demonstration_{num_drones}drones.gif")
    with PygameRecord(gif_path, fps=5) as recorder:
        # Reset do ambiente
        obs = env.reset()
        done = False
        total_reward = 0
        
        # Loop de execução
        while not done:
            # Predizer ação
            action, _ = model.predict(obs, deterministic=True)
            
            # Executar ação
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Gravar frame
            recorder.add_frame()
    
    env.close()
    print(f"GIF salvo em {gif_path} (Recompensa total: {total_reward:.2f})")

if __name__ == "__main__":
    # Testar diretamente para debugging
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_drones", type=int, default=4)
    parser.add_argument("--num_episodes", type=int, default=100000)
    parser.add_argument("--models_dir", default="./models")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ppo_epochs", type=int, default=15)
    parser.add_argument("--plot_freq", type=int, default=10000)
    parser.add_argument("--pre_render_time", type=int, default=2)
    parser.add_argument("--create_gif", action="store_true", default=True)
    parser.add_argument("--prob_matrix_path", default="min_matrix.npy")
    
    args = parser.parse_args()
    
    # Treinar modelo
    train(args)