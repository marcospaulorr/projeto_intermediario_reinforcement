"""
Implementação do modelo de rede neural para o agente PPO
"""

import torch
import torch.nn as nn
import numpy as np

# Abra model.py e substitua a classe PPONetwork por esta versão melhorada
class PPONetwork(nn.Module):
    def __init__(self, obs_shape, action_size):
        super(PPONetwork, self).__init__()
        
        # Arquitetura CNN melhorada para a matriz de probabilidade
        self.conv = nn.Sequential(
            # Primeira camada: mais filtros e menor kernel
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Segunda camada
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Terceira camada 
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Pooling controlado (não tão agressivo)
            nn.MaxPool2d(kernel_size=2),
            
            # Camada final
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        # Calcular tamanho da saída do conv adaptando ao tamanho de entrada
        h, w = obs_shape
        h_out = h // 2  # Devido ao MaxPool2d com kernel_size=2
        w_out = w // 2
        conv_output_size = 32 * h_out * w_out
        
        # MLP para posição (ampliado)
        self.fc_pos = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Camadas comuns (mais profundas)
        self.fc_common = nn.Sequential(
            nn.Linear(conv_output_size + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Cabeça de política
        self.policy = nn.Sequential(
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Cabeça de valor
        self.value = nn.Linear(128, 1)
    
    def forward(self, prob_matrix, position):
        # Processar matriz de probabilidade
        x1 = prob_matrix.unsqueeze(1)  # [B, 1, H, W]
        x1 = self.conv(x1)
        
        # Processar posição
        x2 = self.fc_pos(position)
        
        # Concatenar características
        x = torch.cat([x1, x2], dim=1)
        x = self.fc_common(x)
        
        # Calcular saídas de política e valor
        policy = self.policy(x)
        value = self.value(x)
        
        return policy, value