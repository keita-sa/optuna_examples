import optuna

study = optuna.load_study(
    # storage="sqlite:///optuna-prepared.db",  # リスト2.16が未実行の場合、この行のコメントを解除する。
    storage="sqlite:///optuna.db",  # リスト2.16が未実行の場合、この行のコメントアウトする。
    study_name="ch2-conditional",
)

df = study.trials_dataframe()
print(df)

df = df.sort_values("value", ascending=False)
print(df[["value", "params_clf"]].head())

optuna.visualization.plot_param_importances(
    study=study,
    params=["gb_max_depth", "gb_min_samples_split"]
).show()

importances = optuna.importance.get_param_importances(
    study=study,
    params=["gb_max_depth", "gb_min_samples_split"]
)

for key, value in importances.items():
    print(f"{key}: {value}")

optuna.visualization.plot_contour(
    study=study,
    params=["gb_max_depth", "gb_min_samples_split"]
).show()