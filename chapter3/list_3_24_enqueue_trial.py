import optuna

def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    y = trial.suggest_float("y", -1, 1)
    return x * y

study = optuna.create_study()

# Added both x and y into the queue
study.enqueue_trial({"x": 0.5, "y": -0.3})

# Added only x into the queue, and y is going to be selected as a normal by Optuna sampler
study.enqueue_trial({"x": 0.9})

# Optimizing and show results
study.optimize(objective, n_trials=3)
for trial in study.trials:
    print(f"[{trial.number}] params={trial.params}, value={trial.value}")