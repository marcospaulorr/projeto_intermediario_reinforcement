#!/usr/bin/env python3
import numpy as np
from DSSE import CoverageDroneSwarmSearch

class EnhancedRewardWrapper:
    def __init__(self, env: CoverageDroneSwarmSearch,
                 prob_weight=10.0, revisit_penalty=0.1,
                 coverage_bonus=5.0, obstacle_mask=None,
                 obstacle_penalty=1.0):
        self.env = env
        self.prob_weight = prob_weight
        self.revisit_penalty = revisit_penalty
        self.coverage_bonus = coverage_bonus
        self.obstacle_penalty = obstacle_penalty
        self.obstacles = obstacle_mask
        self.visited_positions = {}
        self.last_cov = 0.0
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.grid_size = env.grid_size
        self.probability_matrix = env.probability_matrix
        self.render_mode = env.render_mode

    @property
    def agents(self): return self.env.agents

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.visited_positions = {a:set() for a in self.env.agents}
        self.last_cov = 0.0
        return obs, info

    # Abra reward_wrapper.py e modifique o método step conforme abaixo
    def step(self, actions):
        obs, rewards, terms, truns, infos = self.env.step(actions)
        pos_after = {aid:tuple(obs[aid][0]) for aid in obs}
        pm = self.env.probability_matrix.get_matrix()
        enhanced = {}
        
        # Para cada agente
        for aid in rewards.keys():
            # Penalidade base por passo (incentiva eficiência)
            enhanced[aid] = -0.05
            
            # Verificar célula atual
            x, y = pos_after[aid]
            
            # PRINCIPAL MUDANÇA: Bônus MUITO maior por visitar novas células com probabilidade
            if pm[y, x] > 0 and pos_after[aid] not in self.visited_positions.get(aid, set()):
                enhanced[aid] += pm[y, x] * self.prob_weight * 10.0  # Multiplicando por 10
            
            # Reduzir penalidade por revisitas
            elif pos_after[aid] in self.visited_positions.get(aid, set()):
                enhanced[aid] -= self.revisit_penalty * 0.5
        
        # Bônus de cobertura global (importante!)
        for info in infos.values():
            if 'coverage_rate' in info:
                cur = info['coverage_rate']
                inc = max(0, cur - self.last_cov)
                if inc > 0:  # Se houve aumento na cobertura
                    bonus = inc * self.coverage_bonus * 5.0  # Multiplicando por 5
                    for aid in enhanced:
                        enhanced[aid] += bonus/len(enhanced)
                self.last_cov = cur
                break
        
        # Atualizar células visitadas
        for aid, pos in pos_after.items():
            self.visited_positions.setdefault(aid, set()).add(pos)
        
        return obs, enhanced, terms, truns, infos
    def render(self): return self.env.render()
    def close(self): return self.env.close()
    def save_matrix(self,path): return self.env.save_matrix(path)