#!/usr/bin/env python3
"""
Script para treinar os agentes PPO no ambiente DSSE Coverage
Versão corrigida com base em implementações bem-sucedidas
"""

import os
import numpy as np
import torch
import gym
from tqdm import tqdm
from DSSE import CoverageDroneSwarmSearch
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from utils import create_directories, plot_metrics
from recorder import PygameRecord

class PrintCallback(BaseCallback):
    """Callback para imprimir progresso durante o treinamento"""
    def __init__(self, print_freq=10000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print(f"[TREINO] Timestep atual: {self.num_timesteps}")
        return True

class DSSECoverageEnv(gym.Env):
    """
    Adapta o ambiente DSSE Coverage para a interface de Gym,
    permitindo uso direto com biblioteca stable-baselines3
    """
    def __init__(self, drone_amount=2, render_mode="ansi", **env_kwargs):
        super().__init__()
        # Cria o ambiente base
        self.env = CoverageDroneSwarmSearch(
            drone_amount=drone_amount, 
            render_mode=render_mode, 
            **env_kwargs, 
            timestep_limit=200
        )
        self.num_drones = drone_amount
        
        # Reset inicial para obter dimensões
        obs_dict, _ = self.env.reset()
        first_agent = list(obs_dict.keys())[0]
        _, prob = obs_dict[first_agent]
        H, W = prob.shape
        self.H, self.W = H, W
        
        # Define posições estratégicas (cantos ou células de maior probabilidade)
        prob_matrix = self.env.probability_matrix.get_matrix()
        valid_cells = [(x, y) for y in range(H) for x in range(W) if prob_matrix[y, x] > 0]
        
        if len(valid_cells) >= drone_amount:
            # Ordena por probabilidade e usa as maiores
            valid_cells.sort(key=lambda pos: prob_matrix[pos[1], pos[0]], reverse=True)
            self.initial_positions = valid_cells[:drone_amount]
        else:
            # Usa os cantos ou posições espaçadas como backup
            corners = [(0, 0), (W-1, 0), (0, H-1), (W-1, H-1)]
            if drone_amount <= 4:
                self.initial_positions = corners[:drone_amount]
            else:
                # Distribuir uniformemente
                self.initial_positions = []
                step_x = max(1, W // int(np.sqrt(drone_amount)))
                step_y = max(1, H // int(np.sqrt(drone_amount)))
                for i in range(min(drone_amount, (W//step_x) * (H//step_y))):
                    x = (i % (W//step_x)) * step_x + step_x//2
                    y = (i // (W//step_x)) * step_y + step_y//2
                    self.initial_positions.append((min(x, W-1), min(y, H-1)))
                # Preencher posições restantes
                while len(self.initial_positions) < drone_amount:
                    self.initial_positions.append(self.initial_positions[0])  # Duplicar primeira posição
        
        # Reset com posições iniciais
        obs_dict, _ = self.env.reset(options={"drones_positions": self.initial_positions})
        
        # Prepara observação achatada
        combined_obs = self._combine_obs(obs_dict)
        
        # Define espaços
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(combined_obs.shape[0],), 
            dtype=np.float32
        )
        
        # Ações discretas para cada drone
        act_n = self.env.action_space(first_agent).n
        self.action_space = spaces.MultiDiscrete([act_n] * self.num_drones)
        
        # Rastreamento interno
        self.visited = np.zeros((H, W), dtype=bool)
        self.last_coverage = 0
        self.current_obs = combined_obs
        
    def _combine_obs(self, obs_dict):
        """Combina observações de todos os drones em um único vetor"""
        agents = sorted(obs_dict.keys())
        pos_list = []
        
        # Extrair posições
        for agent in agents:
            (x, y), _ = obs_dict[agent]
            pos_list.extend([float(x), float(y)])
            
        # Usar a matriz de probabilidade do primeiro drone (deve ser a mesma para todos)
        _, prob_matrix = obs_dict[agents[0]]
        flat_prob = prob_matrix.flatten().astype(np.float32)
        
        return np.concatenate([np.array(pos_list, dtype=np.float32), flat_prob])
    
    def reset(self):
        """Reset do ambiente com posições iniciais estratégicas"""
        obs_dict, _ = self.env.reset(options={"drones_positions": self.initial_positions})
        self.current_obs = self._combine_obs(obs_dict)
        self.visited = np.zeros((self.H, self.W), dtype=bool)
        self.last_coverage = 0
        return self.current_obs
    
    def step(self, action):
        """Processa ação e aplica lógica de recompensa aprimorada"""
        # Converte para o formato esperado pelo ambiente base
        act_dict = {}
        for i in range(min(len(action), self.num_drones)):
            drone_id = f"drone{i}"
            if drone_id in self.env.agents:
                act_dict[drone_id] = int(action[i])
        
        # Executa passo no ambiente
        obs_dict, rewards, term, trunc, infos = self.env.step(act_dict)
        
        # Calcula recompensa com bônus/penalidades
        reward = sum(rewards.values())
        
        # Penalidade por revisita
        revisit_penalty = 0
        for agent in obs_dict:
            (x, y), _ = obs_dict[agent]
            if self.visited[y, x]:
                revisit_penalty -= 0.2
            self.visited[y, x] = True
        
        # Bônus por cobertura
        coverage_bonus = 0
        for info in infos.values():
            if 'coverage_rate' in info:
                current_cov = info['coverage_rate']
                if current_cov > self.last_coverage:
                    coverage_bonus += (current_cov - self.last_coverage) * 5.0
                self.last_coverage = current_cov
                break
        
        # Recompensa final
        final_reward = reward + revisit_penalty + coverage_bonus
        
        # Verificar término
        done = not self.env.agents or all(term.get(a, False) or trunc.get(a, False) for a in self.env.agents)
        
        # Informações adicionais
        info = {
            "drones_info": infos,
            "revisit_penalty": revisit_penalty,
            "coverage_bonus": coverage_bonus,
            "coverage_rate": self.last_coverage
        }
        
        # Atualiza observação atual
        self.current_obs = None if done else self._combine_obs(obs_dict)
        
        return self.current_obs, final_reward, done, info
    
    def render(self, mode="human"):
        try:
            return self.env.render()
        except:
            return None
        
    def close(self):
        try:
            self.env.close()
        except:
            pass


def train(args, experiment_config=None):
    """Treina um agente PPO no ambiente DSSE Coverage usando stable-baselines3"""
    print(f"Iniciando treinamento com {args.num_drones} drones")
    
    # Criar diretórios para salvar modelos e resultados
    exp_name = args.experiment if args.experiment else "default"
    models_dir = args.models_dir
    logs_dir = os.path.join(models_dir, 'logs')
    create_directories([models_dir, logs_dir])
    
    # Configurar logger
    logger = configure(logs_dir, ["stdout", "csv", "tensorboard"])
    
    # Criar ambiente
    env = DSSECoverageEnv(
        drone_amount=args.num_drones, 
        render_mode="ansi",
        disaster_position=(-24.04, -46.17),
        pre_render_time=args.pre_render_time
    )
    
    # Definir hiperparâmetros do PPO
    if experiment_config:
        learning_rate = experiment_config["learning_rate"]
        gamma = experiment_config["gamma"]
        clip_ratio = experiment_config["clip_ratio"]
        ent_coef = experiment_config["entropy_coef"]
    else:
        learning_rate = args.learning_rate
        gamma = args.gamma
        clip_ratio = 0.2
        ent_coef = 0.01
    
    # Criar modelo PPO com SB3
    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=learning_rate,
        n_steps=1024,             
        batch_size=args.batch_size,           
        n_epochs=args.ppo_epochs,              
        gamma=gamma,               
        gae_lambda=0.95,          
        clip_range=clip_ratio,          
        ent_coef=ent_coef,            
        vf_coef=0.5,              
        max_grad_norm=0.5,
        tensorboard_log=logs_dir,
        verbose=1
    )
    
    # Definir callback para progresso
    callback = PrintCallback(print_freq=args.plot_freq)
    
    # Treinar o modelo
    print(f"Treinando modelo por {args.num_episodes} timesteps...")
    model.learn(
        total_timesteps=args.num_episodes,
        callback=callback,
        tb_log_name=exp_name
    )
    
    # Salvar modelo
    model_path = os.path.join(models_dir, f"ppo_{exp_name}_final")
    model.save(model_path)
    print(f"Modelo salvo em {model_path}")
    
    # Fechar ambiente
    env.close()
    
    # Opcional: criar GIF demonstrativo
    if args.create_gif:
        create_demonstration_gif(model, args.num_drones, models_dir, exp_name)
    
    return model


def create_demonstration_gif(model, num_drones, save_dir, exp_name="default"):
    """Cria um GIF demonstrando o comportamento do agente treinado"""
    print("Criando GIF de demonstração...")
    
    # Criar ambiente de avaliação com render visual
    env = DSSECoverageEnv(
        drone_amount=num_drones,
        render_mode="human",
        disaster_position=(-24.04, -46.17),
        pre_render_time=2
    )
    
    # Preparar gravação
    gif_path = os.path.join(save_dir, f"demonstration_{exp_name}.gif")
    with PygameRecord(gif_path, fps=5) as recorder:
        obs = env.reset()
        done = False
        
        while not done:
            # Predizer ação
            action, _ = model.predict(obs, deterministic=True)
            
            # Executar ação
            obs, _, done, _ = env.step(action)
            
            # Gravar frame
            recorder.add_frame()
    
    env.close()
    print(f"GIF salvo em {gif_path}")