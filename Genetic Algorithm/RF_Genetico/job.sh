#!/bin/bash
#SBATCH --job-name=MLP_AG
#SBATCH --output=saida_%j.txt
#SBATCH --error=erro_%j.txt
#SBATCH --partition=nanotubo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G

# Ativa o ambiente virtual usando caminho absoluto
source /home/santos.arthur/Nuno/meuambiente/bin/activate

# Roda o script com saída não-bufferizada
python -u Algoritmo_genetico.py
