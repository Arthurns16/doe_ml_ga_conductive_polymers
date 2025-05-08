import joblib
import pickle
import numpy as np
import pandas as pd
import random
import math
from collections import Counter
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Fix de semente para reprodutibilidade
random.seed(42)
np.random.seed(42)

# 1) Carregar modelo e scaler
model = joblib.load("best_mlp_model.joblib")
if hasattr(model, "n_jobs"):
    model.n_jobs = 1

with open("escala.pickle", "rb") as f:
    scaler = pickle.load(f)

# 2) Ordem das features e índices
feat_names = [
    "SW","MW","GR",
    "RPM","Tempo (minutos)",
    "PA66","PC","PEI","PPA","PVDF"
]
idx_g1   = range(0,3)
idx_rpm  = 3
idx_time = 4
idx_g2   = range(5,10)
N_FEAT   = len(feat_names)

# 3) Geração e reparo de indivíduo
def generate_individual():
    ind = np.zeros(N_FEAT)
    ind[random.choice(idx_g1)] = 1
    ind[random.choice(idx_g2)] = 1
    ind[idx_rpm]  = random.uniform(0,400)
    ind[idx_time] = random.uniform(0,20)
    return ind

def repair_individual(ind):
    if ind[list(idx_g1)].sum() != 1:
        ind[list(idx_g1)] = 0
        ind[random.choice(idx_g1)] = 1
    if ind[list(idx_g2)].sum() != 1:
        ind[list(idx_g2)] = 0
        ind[random.choice(idx_g2)] = 1
    ind[idx_rpm]  = np.clip(ind[idx_rpm],  0, 400)
    ind[idx_time] = np.clip(ind[idx_time], 0, 20)
    return ind

# 4) Função de aptidão — usa DataFrame apenas para o scaler
def fitness(ind):
    df = pd.DataFrame(ind.reshape(1, -1), columns=feat_names)
    scaled_arr = scaler.transform(df)
    return model.predict(scaled_arr)[0]

# 5) Crossover (mantém regras one-hot + blend)
def crossover(p1, p2):
    c1, c2 = np.zeros(N_FEAT), np.zeros(N_FEAT)
    # one-hot carga
    i1 = (np.where(p1[list(idx_g1)] == 1)[0][0]
          if random.random()<0.5
          else np.where(p2[list(idx_g1)] == 1)[0][0])
    c1[i1] = 1
    i2 = (np.where(p2[list(idx_g1)] == 1)[0][0]
          if random.random()<0.5
          else np.where(p1[list(idx_g1)] == 1)[0][0])
    c2[i2] = 1

    # one-hot polímero
    j1 = (np.where(p1[list(idx_g2)] == 1)[0][0]
          if random.random()<0.5
          else np.where(p2[list(idx_g2)] == 1)[0][0])
    c1[j1 + idx_g2.start] = 1
    j2 = (np.where(p2[list(idx_g2)] == 1)[0][0]
          if random.random()<0.5
          else np.where(p1[list(idx_g2)] == 1)[0][0])
    c2[j2 + idx_g2.start] = 1

    # blend linear em contínuas
    alpha, beta = random.random(), random.random()
    c1[idx_rpm]  = alpha * p1[idx_rpm]  + (1-alpha) * p2[idx_rpm]
    c1[idx_time] = beta  * p1[idx_time] + (1-beta)  * p2[idx_time]
    c2[idx_rpm]  = (1-alpha) * p1[idx_rpm]  + alpha * p2[idx_rpm]
    c2[idx_time] = (1-beta)  * p1[idx_time] + beta  * p2[idx_time]

    return repair_individual(c1), repair_individual(c2)

# 6) Mutação categórica e contínua
def mutate(ind, mut_cat, mut_cont):
    if random.random() < mut_cat:
        curr = list(idx_g1)[np.where(ind[list(idx_g1)] == 1)[0][0]]
        ind[curr] = 0
        ind[random.choice(idx_g1)] = 1
    if random.random() < mut_cat:
        curr = list(idx_g2)[np.where(ind[list(idx_g2)] == 1)[0][0]]
        ind[curr] = 0
        ind[random.choice(idx_g2)] = 1
    if random.random() < mut_cont:
        ind[idx_rpm]  = random.uniform(0, 400)
    if random.random() < mut_cont:
        ind[idx_time] = random.uniform(0, 20)
    return repair_individual(ind)

# 7) Seleção por torneio
def tournament_selection(pop, fits, k):
    winners = []
    for _ in pop:
        aspirants = random.sample(list(zip(pop, fits)), k)
        winners.append(max(aspirants, key=lambda x: x[1])[0])
    return winners

# 8) Parâmetros do GA
POP_SIZE       = 12032
N_GEN          = 1197
ELIT_FRAC      = 0.01
ELITE          = max(1, int(POP_SIZE * ELIT_FRAC))
TOURN_K        = 2
N_JOBS         = 1
MUT_MIN_CAT    = 0.20
MUT_MAX_CAT    = 0.40
MUT_MIN_CONT   = 0.02
MUT_MAX_CONT   = 0.20
IMMIGRANT_FRAC = 0.04  # 4%

# 9) Históricos
elite_history   = []
all_history     = []
best_history    = []
avg_history     = []
mut_cat_hist    = []
mut_cont_hist   = []
diversity_hist  = []

# 10) População inicial
pop = [generate_individual() for _ in range(POP_SIZE)]

# 11) Loop evolutivo
for gen in range(1, N_GEN+1):
    # avalia em paralelo
    fits = Parallel(n_jobs=N_JOBS)(delayed(fitness)(ind) for ind in pop)
    best = max(fits)
    avg  = sum(fits)/len(fits)

    # diversidade via Shannon (categórico + bins contínuos)
    n_bins_rpm, n_bins_time = 10, 10
    rpm_vals  = [ind[idx_rpm]  for ind in pop]
    time_vals = [ind[idx_time] for ind in pop]
    rpm_bins  = np.digitize(rpm_vals,  bins=np.linspace(0,400,n_bins_rpm+1)) - 1
    time_bins = np.digitize(time_vals, bins=np.linspace(0,20, n_bins_time+1)) - 1

    combo_list = [
        (
          np.where(ind[list(idx_g1)]==1)[0][0],
          np.where(ind[list(idx_g2)]==1)[0][0],
          rb, tb
        )
        for ind, rb, tb in zip(pop, rpm_bins, time_bins)
    ]
    counts = Counter(combo_list)
    probs  = [c/POP_SIZE for c in counts.values()]
    H      = -sum(p * math.log(p) for p in probs)
    H_max  = math.log(len(idx_g1) * len(idx_g2) * n_bins_rpm * n_bins_time)
    diversity = H / H_max

    # adaptar taxas
    mut_cat  = MUT_MIN_CAT  + (1-diversity)*(MUT_MAX_CAT - MUT_MIN_CAT)
    mut_cont = MUT_MIN_CONT + (1-diversity)*(MUT_MAX_CONT - MUT_MIN_CONT)

    # armazenar históricos
    best_history.append(best)
    avg_history.append(avg)
    mut_cat_hist.append(mut_cat)
    mut_cont_hist.append(mut_cont)
    diversity_hist.append(diversity)
    ranked = sorted(zip(pop, fits), key=lambda x: x[1], reverse=True)
    elite_history.append(ranked[:ELITE])
    all_history.append(list(zip(pop, fits)))

    print(
        f"G{gen:03d} → best={best:.4f}, avg={avg:.4f}, "
        f"H_norm={diversity:.2f}, mut_cat={mut_cat:.2f}, mut_cont={mut_cont:.2f}"
    )

    # 11.1) Elitismo
    hard_elites = [ind.copy() for ind,_ in ranked[:ELITE]]
    # 11.2) Seleção de pais
    mating_pool = tournament_selection(pop, fits, TOURN_K)

    # 11.3) Criação da próxima geração
    newpop, i = hard_elites.copy(), 0
    while len(newpop) < POP_SIZE:
        p1, p2 = mating_pool[i % len(mating_pool)], mating_pool[(i+1) % len(mating_pool)]
        c1, c2 = crossover(p1, p2)
        newpop.extend([
            mutate(c1, mut_cat, mut_cont),
            mutate(c2, mut_cat, mut_cont)
        ])
        i += 2

    # 11.4) Imigrantes aleatórios
    n_imm = max(1, int(POP_SIZE * IMMIGRANT_FRAC))
    for idx in random.sample(range(ELITE, POP_SIZE), n_imm):
        newpop[idx] = generate_individual()

    pop = newpop[:POP_SIZE]

# 12) Salvar métricas em pickle
with open("history_metrics.pkl", "wb") as f:
    pickle.dump({
        "best_history":    best_history,
        "avg_history":     avg_history,
        "mut_cat_hist":    mut_cat_hist,
        "mut_cont_hist":   mut_cont_hist,
        "diversity_hist":  diversity_hist
    }, f)

# 13) Gerar e salvar gráfico (sem exibir)
fig, axes = plt.subplots(1, 3, figsize=(15,5))
axes[0].plot(best_history, label="Melhor")
axes[0].plot(avg_history,  label="Média")
axes[0].set_title("Fitness");     axes[0].legend()
axes[1].plot(mut_cat_hist, label="mut_cat")
axes[1].plot(mut_cont_hist,label="mut_cont")
axes[1].set_title("Taxas");       axes[1].legend()
axes[2].plot(diversity_hist, label="Índice de Shannon")
axes[2].set_title("Diversidade"); axes[2].legend()
plt.tight_layout()
plt.savefig("history_metrics.png")
plt.close(fig)

# 14) Salvar última geração completa em Excel
last_gen = all_history[-1]
records_last = []
for idx, (ind, fit) in enumerate(last_gen):
    rec = {"Ind": idx, "Fitness": fit}
    rec.update({feat_names[i]: ind[i] for i in range(N_FEAT)})
    records_last.append(rec)
pd.DataFrame(records_last).to_excel("ultima_geracao.xlsx", index=False)

print("Histórico salvo em 'history_metrics.pkl', gráfico em 'history_metrics.png' e última geração em 'ultima_geracao.xlsx'")
