import optuna
from binh_and_korn import objective

def f1(x, y):
    return 4 * x**2 + 4 * y**2

def f2(x, y):
    return (x - 5)**2 + (y - 5)**2

def objective(trial):
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)

    v1 = f1(x, y)
    v2 = f2(x, y)

    return v1, v2

study = optuna.create_study(directions=["minimize", "minimize"])
study.optimize(objective, n_trials=1000)

print("[Best Trials]")

for trial in study.best_trials:
    print(f"- [{trial.number}] params={trial.params}, values={trial.values}")

study = optuna.load_study(
    study_name="ch3-multi-objective-example",
    storage="sqlite:///optuna.db"
)

# すべてのトライアルをプロット（デフォルト挙動）
optuna.visualization.plot_pareto_front(
	study,
	include_dominated_trials=True
).show()

# Study.best_trials だけをプロット
optuna.visualization.plot_pareto_front(
	study,
	include_dominated_trials=False
).show()