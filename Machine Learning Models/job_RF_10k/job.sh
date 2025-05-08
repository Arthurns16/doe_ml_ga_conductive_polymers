#!/bin/bash
#SBATCH --job-name=randomforest_max
#SBATCH --output=saida_%j.txt
#SBATCH --error=erro_%j.txt
#SBATCH --partition=nanotubo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G

# Ativa o ambiente virtual (ajuste o caminho se necessário)
source meuambiente/bin/activate

# Roda o script com saída não-bufferizada
python -u Random_Forest_unificado.py

