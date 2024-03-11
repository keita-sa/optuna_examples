import optuna
from optuna.visualization import plot_optimization_history

study = optuna.load_study(
    study_name="ch3-lightgbm-search-space-v3-tpe",
    storage="sqlite:///optuna.db",
)

plot_optimization_history(study).show()
