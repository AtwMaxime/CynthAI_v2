# Protocol — Avant de lancer quoi que ce soit

## 1. Se connecter à auriga

```bash
ssh auriga
```

---

## 2. Vérifier l'état des GPUs

```bash
nvidia-smi
```

Points à vérifier :
- **GPU-Util** : si les deux GPUs sont à ~100%, quelqu'un tourne déjà quelque chose — attendre ou coordonner
- **Memory-Usage** : chaque RTX A6000 a 49 140 MiB. S'assurer qu'il reste assez de place (l'entraînement CynthAI utilise ~4-8 GB selon la config)
- **Processes** : identifier qui tourne quoi

---

## 3. Naviguer dans le projet

```bash
cd /local_scratch/mattwood/projects/rl_agent/CynthAI_v2
```

---

## 4. Activer le venv

```bash
source .venv/bin/activate
```

Vérifier que tout est OK :

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "from simulator import PyBattle; print('Simulator OK')"
```

Résultat attendu :
```
2.12.0+cu130
CUDA: True
Simulator OK
```

---

## 5. Lancer l'entraînement

```bash
# Mode recommandé — cheater + independent critic
python run_cheater_indep_critic.py

# Ou curriculum
python run_curriculum_max.py

# Reprendre depuis un checkpoint
python run_cheater_indep_critic.py --resume checkpoints/<run_name>/agent_NNNNNN.pt
```

Pour garder la session active après déconnexion, utiliser **tmux** :

```bash
tmux new -s train          # créer une session nommée "train"
python run_cheater_indep_critic.py
# Ctrl+B puis D pour détacher sans tuer le process
```

Se rattacher plus tard :

```bash
tmux attach -t train
```

Lister les sessions actives :

```bash
tmux ls
```

---

## 6. Lancer les tests

```bash
python tests/run_all_tests.py
```

4 suites, ~100 assertions. Tout doit passer avant de lancer un entraînement.

---

## 7. Rebuild du bridge Rust si nécessaire

Nécessaire après un `git pull` qui modifie `simulator/src/lib.rs` ou `pokemon-showdown-rs`.

```bash
cd /local_scratch/mattwood/projects/rl_agent/CynthAI_v2/simulator
source ../.venv/bin/activate
unset CONDA_PREFIX  # important sur auriga (conda est dans l'env par défaut)
maturin develop --release
cd ..
```

---

## 8. Synchroniser le code depuis la machine locale

Depuis la machine locale (Windows), dans Git Bash :

```bash
scp -r /c/Users/Maxime/Desktop/PokemonAI/CynthAI_v2 /c/Users/Maxime/Desktop/PokemonAI/pokemon-showdown-rs auriga:/local_scratch/mattwood/projects/rl_agent/
```

---

## Notes

- **CONDA_PREFIX** est souvent défini sur auriga par défaut — `unset CONDA_PREFIX` avant `maturin develop` sinon il plante
- Le venv est dans `CynthAI_v2/.venv/` — ne pas utiliser le python système
- Les checkpoints ne sont pas synchronisés (exclus du scp) — ils restent sur auriga
- Les deux GPUs (0 et 1) sont disponibles. PyTorch prend le GPU 0 par défaut. Pour forcer un GPU : `CUDA_VISIBLE_DEVICES=1 python run_cheater_indep_critic.py`
