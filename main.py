#!/usr/bin/env python3
"""
Script principal para o projeto de Coverage Path Planning com RL
"""

import os
import argparse
from train import train
from evaluate import compare_algorithms
from utils import create_directories

# Definições de experimentos
# Abra main.py e substitua a definição de EXPERIMENTS por esta versão melhorada
EXPERIMENTS = {
    "default": {
        "gamma": 0.95,       # Reduzido de 0.99 
        "clip_ratio": 0.2,
        "entropy_coef": 0.05, # Aumentado de 0.01
        "learning_rate": 1e-3 # Aumentado de 3e-4
    },
    "exp1_exploration": {
        "gamma": 0.9,        # Foco maior em recompensas imediatas
        "clip_ratio": 0.3,   # Permite atualizações maiores
        "entropy_coef": 0.1,  # Alta exploração
        "learning_rate": 1e-3
    },
    "exp2_longterm": {
        "gamma": 0.98,
        "clip_ratio": 0.2,
        "entropy_coef": 0.05,
        "learning_rate": 5e-4
    },
    "exp3_balanced": {
        "gamma": 0.95,
        "clip_ratio": 0.25,
        "entropy_coef": 0.03,
        "learning_rate": 7e-4
    },
    # Nova configuração com exploração extrema
    "exp4_high_explore": {
        "gamma": 0.9,
        "clip_ratio": 0.3,
        "entropy_coef": 0.2,  # Exploração muito alta
        "learning_rate": 2e-3  # Learning rate agressivo
    }
}



def main(args):
    """Função principal para executar o projeto"""
    # Criar diretórios necessários
    create_directories([args.models_dir, args.results_dir])
    
    # Carregar configuração de experimento
    experiment_config = None
    if args.experiment:
        if args.experiment in EXPERIMENTS:
            experiment_config = EXPERIMENTS[args.experiment]
            print(f"Usando configuração de experimento: {args.experiment}")
            print(f"Hiperparâmetros: {experiment_config}")
            
            # Criar subdiretório para este experimento
            exp_dir = os.path.join(args.models_dir, args.experiment)
            create_directories([exp_dir])
            args.models_dir = exp_dir
        else:
            print(f"Experimento '{args.experiment}' não encontrado. Usando configuração padrão.")
    
    # MODO TRAIN
    if args.mode in ['train', 'both']:
        print("=== Iniciando treinamento ===")
        train(args, experiment_config)
    
    # MODO EVALUATE (agora somente PPO)
    if args.mode in ['evaluate', 'both']:
        print("=== Iniciando avaliação (PPO) ===")
        num_eval = 500 if args.long_eval else args.eval_episodes
        compare_algorithms(
            agents_dir=args.models_dir,
            num_drones=args.num_drones,
            num_episodes=num_eval,
            render=args.render,
            results_dir=args.results_dir
        )
        print(f"Avaliação concluída. Resultados salvos em {args.results_dir}")
    
    print("Execução concluída!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Coverage Path Planning com Reinforcement Learning"
    )
    parser.add_argument("--mode", choices=["train","evaluate","both"], default="both",
                        help="Modo de execução (train, evaluate, both)")
    parser.add_argument("--num_drones", type=int, default=2,
                        help="Número de drones no ambiente")
    parser.add_argument("--models_dir", default="./models",
                        help="Diretório para salvar/carregar modelos")
    parser.add_argument("--results_dir", default="./results",
                        help="Diretório para salvar resultados")
    parser.add_argument("--experiment", choices=list(EXPERIMENTS.keys()), default=None,
                        help="Configuração de experimento")
    parser.add_argument("--num_episodes", type=int, default=2000,
                        help="Número de episódios de treinamento")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Taxa de aprendizado")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Fator de desconto")
    parser.add_argument("--ppo_epochs", type=int, default=10,
                        help="Épocas PPO")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--save_freq", type=int, default=100,
                        help="Frequência de salvamento")
    parser.add_argument("--plot_freq", type=int, default=50,
                        help="Frequência de plotagem")
    parser.add_argument("--pre_render_time", type=int, default=2,
                        help="Pre-render time")
    parser.add_argument("--create_gif", action="store_true",
                        help="Criar GIF no final")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Desativar CUDA")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Episódios de avaliação padrão")
    parser.add_argument("--long_eval", action="store_true",
                        help="Avaliação longa (500 episódios)")
    parser.add_argument("--render", action="store_true",
                        help="Renderizar durante avaliação")

    args = parser.parse_args()
    main(args)
