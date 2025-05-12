#!/usr/bin/env python3
"""
Script para executar múltiplas execuções do PPO e gerar gráficos comparativos
"""

import os
import numpy as np
import pandas as pd
import argparse
from utils import create_directories, plot_multiple_runs
from main import main, EXPERIMENTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executa múltiplas rodadas de treinamento e avaliação"
    )
    parser.add_argument("--experiment", choices=list(EXPERIMENTS.keys()), default="default",
                        help="Configuração de experimento")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Número de execuções")
    parser.add_argument("--num_drones", type=int, default=2,
                        help="Número de drones")
    parser.add_argument("--num_episodes", type=int, default=50000,
                        help="Episódios por execução")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Semente base para gerar sementes aleatórias")
    args = parser.parse_args()
    
    # Configurar diretórios
    base_dir = f"./multi_run_{args.experiment}"
    create_directories([base_dir])
    
    # Lista para armazenar resultados de cada execução
    all_results = []
    
    # Executar múltiplas vezes
    for run in range(args.num_runs):
        print(f"\n===== EXECUÇÃO {run+1}/{args.num_runs} =====")
        
        # Criar argumentos para esta execução
        run_args = argparse.Namespace(
            mode="both",
            num_drones=args.num_drones,
            models_dir=f"{base_dir}/models_run{run}",
            results_dir=f"{base_dir}/results_run{run}",
            experiment=args.experiment,
            num_episodes=args.num_episodes,
            learning_rate=EXPERIMENTS[args.experiment]["learning_rate"],
            gamma=EXPERIMENTS[args.experiment]["gamma"],
            ppo_epochs=10,
            batch_size=128,
            save_freq=100,
            plot_freq=500,
            pre_render_time=2,
            create_gif=False,
            no_cuda=False,
            eval_episodes=5,
            long_eval=False,
            render=False,
            seed=args.base_seed + run
        )
        
        # Executar treinamento e avaliação
        main(run_args)
        
        # Carregar resultados
        results_csv = f"{base_dir}/results_run{run}/detailed_results.csv"
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            # Organizar por algoritmo e métrica
            result_dict = {"PPO": {}}
            for metric in ['reward', 'coverage_rate', 'repeated_coverage', 'cumulative_pos']:
                metric_data = df[df['metric'] == metric]
                if not metric_data.empty:
                    if 'step' in metric_data.columns:
                        # Ordenar por passo
                        metric_data = metric_data.sort_values('step')
                        result_dict["PPO"][metric] = metric_data['value'].tolist()
            
            all_results.append(result_dict)
    
    # Após todas as execuções, gerar gráficos comparativos
    print("\n===== Gerando gráficos comparativos =====")
    for metric in ['reward', 'coverage_rate', 'repeated_coverage', 'cumulative_pos']:
        plot_multiple_runs(
            all_results, 
            "PPO", 
            metric, 
            f"{base_dir}/combined_{metric}.png"
        )
    
    print(f"\nResultados salvos em {base_dir}")