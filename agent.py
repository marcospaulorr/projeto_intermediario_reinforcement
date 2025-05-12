"""
Implementação do agente PPO para o problema de Coverage Path Planning
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from model import PPONetwork

class PPOAgent:
    def __init__(
        self,
        obs_shape,
        action_size,
        lr=3e-4,
        gamma=0.99,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        device="cpu",
        experiment_config=None
    ):
        self.obs_shape = obs_shape
        self.action_size = action_size
        
        # Aplicar configuração de experimento se fornecida
        if experiment_config:
            self.gamma = experiment_config["gamma"]
            self.clip_ratio = experiment_config["clip_ratio"]
            self.entropy_coef = experiment_config["entropy_coef"]
            lr = experiment_config["learning_rate"]
        else:
            self.gamma = gamma
            self.clip_ratio = clip_ratio 
            self.entropy_coef = entropy_coef
            
        self.value_coef = value_coef
        self.device = device
        
        # Inicialização da rede neural
        self.network = PPONetwork(obs_shape, action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Buffers para armazenar experiências
        self.reset_buffers()
    
    def reset_buffers(self):
        """Limpa os buffers de experiência"""
        self.positions = []
        self.matrices = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def act(self, obs, training=True):
        """Seleciona uma ação com base na observação atual"""
        position, prob_matrix = obs
        position = torch.FloatTensor(position).unsqueeze(0).to(self.device)
        prob_matrix = torch.FloatTensor(prob_matrix).unsqueeze(0).to(self.device)
        
        # Forward pass na rede
        with torch.no_grad():
            policy, value = self.network(prob_matrix, position)
            
        # Amostrar ação da distribuição de política
        if training:
            dist = Categorical(policy)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Armazenar experiência
            self.positions.append(position)
            self.matrices.append(prob_matrix)
            self.actions.append(action)
            self.values.append(value)
            self.log_probs.append(log_prob)
            
            return action.item()
        else:
            # No modo de avaliação, escolher a ação mais provável
            return policy.argmax(dim=1).item()
    
    def remember(self, reward, done):
        """Armazena recompensa e flag de término"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def learn(self, next_obs=None, next_done=None, epochs=10, batch_size=64):
        """Atualiza a política com base nas experiências coletadas"""
        # Converter listas para tensores
        positions = torch.cat(self.positions)
        matrices = torch.cat(self.matrices)
        actions = torch.cat(self.actions)
        old_log_probs = torch.cat(self.log_probs)
        
        # Calcular retornos e vantagens
        returns, advantages = self._compute_returns_advantages(next_obs, next_done)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Treinamento por múltiplas épocas
        for _ in range(epochs):
            # Criar batches
            indices = np.random.permutation(len(returns))
            n_batches = len(returns) // batch_size
            
            for start_idx in range(0, len(returns), batch_size):
                end_idx = min(start_idx + batch_size, len(returns))
                batch_indices = indices[start_idx:end_idx]
                
                # Obter batch de dados
                batch_positions = positions[batch_indices]
                batch_matrices = matrices[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                policy, values = self.network(batch_matrices, batch_positions)
                
                # Calcular probabilidades de ações
                dist = Categorical(policy)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Calcular razão de políticas
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Calcular perda surrogate clipped
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Calcular perda de valor
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                
                # Perda total
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Atualizar pesos
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Limpar buffers após aprendizado
        self.reset_buffers()
    
    def _compute_returns_advantages(self, next_obs, next_done):
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        values = torch.cat(self.values).squeeze(-1)
        
        # Calcular valor do próximo estado
        next_value = 0
        if next_obs is not None:
            next_position, next_matrix = next_obs
            next_position = torch.FloatTensor(next_position).unsqueeze(0).to(self.device)
            next_matrix = torch.FloatTensor(next_matrix).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, next_value = self.network(next_matrix, next_position)
                next_value = next_value.squeeze(-1)
                if next_done:
                    next_value = 0
        
        # Use Generalized Advantage Estimation (GAE)
        returns_list = []  # Lista Python normal, não tensor
        advantages_list = []  # Lista Python normal, não tensor
        advantage = 0
        next_value = float(next_value) if isinstance(next_value, torch.Tensor) else next_value
        gae_lambda = 0.95  # Parâmetro lambda para GAE
        
        for t in reversed(range(len(rewards))):
            # Delta: r + gamma*V(s+1) - V(s)
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            # GAE: A(s,a) = delta + gamma*lambda*A(s+1,a+1)
            advantage = delta + self.gamma * gae_lambda * advantage * next_non_terminal
            returns_list.insert(0, float(advantage + values[t]))  # Convertido para float
            advantages_list.insert(0, float(advantage))  # Convertido para float
            
            next_value = values[t]
        
        # Converter listas para tensores
        returns = torch.FloatTensor(returns_list).to(self.device)
        advantages = torch.FloatTensor(advantages_list).to(self.device)
        
        # Normalizar vantagens
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def save(self, path):
        """Salva o modelo do agente"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_shape': self.obs_shape,
            'gamma': self.gamma,
            'clip_ratio': self.clip_ratio,
            'entropy_coef': self.entropy_coef
        }, path)
    
    def load(self, path):
        """Carrega o modelo salvo com tratamento de erro para dimensões diferentes"""
        try:
            checkpoint = torch.load(path)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Carregar hiperparâmetros se disponíveis
            if 'gamma' in checkpoint:
                self.gamma = checkpoint['gamma']
            if 'clip_ratio' in checkpoint:
                self.clip_ratio = checkpoint['clip_ratio']
            if 'entropy_coef' in checkpoint:
                self.entropy_coef = checkpoint['entropy_coef']
                
            print(f"Modelo carregado com sucesso: {path}")
            print(f"Hiperparâmetros: gamma={self.gamma}, clip_ratio={self.clip_ratio}, entropy_coef={self.entropy_coef}")
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            print("Tentando carregar o modelo com tratamento para incompatibilidade de dimensões...")
            
            checkpoint = torch.load(path)
            saved_state_dict = checkpoint['network_state_dict']
            model_state_dict = self.network.state_dict()
            
            # Copiar parâmetros compatíveis
            for name, param in saved_state_dict.items():
                if name in model_state_dict:
                    if model_state_dict[name].shape == param.shape:
                        model_state_dict[name].copy_(param)
                    else:
                        print(f"Ignorando parâmetro incompatível: {name} (salvo: {param.shape}, atual: {model_state_dict[name].shape})")
            
            # Aplicar parâmetros atualizados
            self.network.load_state_dict(model_state_dict)
            print("Modelo carregado parcialmente com sucesso!")
    
    @classmethod
    def create_from_checkpoint(cls, path, device="cpu"):
        """Cria um agente a partir de um checkpoint salvo"""
        checkpoint = torch.load(path)
        
        # Verificar se o checkpoint contém informações sobre o tamanho da observação
        if 'obs_shape' in checkpoint:
            obs_shape = checkpoint['obs_shape']
        else:
            # Tentar inferir do state_dict
            print("Aviso: Checkpoint não contém informações sobre obs_shape, usando inferência")
            # Valor padrão seguro
            obs_shape = (45, 45)
        
        # Extrair hiperparâmetros do checkpoint se disponíveis
        gamma = checkpoint.get('gamma', 0.99)
        clip_ratio = checkpoint.get('clip_ratio', 0.2)
        entropy_coef = checkpoint.get('entropy_coef', 0.01)
        
        # Criar agente com o tamanho correto e hiperparâmetros
        agent = cls(
            obs_shape=obs_shape,
            action_size=8,  # Fixado em 8 ações para o problema CPP
            gamma=gamma,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            device=device
        )
        
        # Tentar carregar o modelo
        agent.load(path)
        
        return agent