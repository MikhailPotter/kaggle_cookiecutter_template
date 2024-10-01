import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error, root_mean_squared_error, recall_score
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb

class BaseModelTuner:
    def __init__(self, X, y, n_trials, metric, directions=['maximize'], cat_features=None):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.metric = metric
        self.directions = directions
        self.cat_features = []
        if cat_features:
            self.cat_features = cat_features

    def objective(self, trial):
        kf = KFold(n_splits=4)
        val_scores, train_scores, diff_scores = [], [], []

        for train_idx, val_idx in kf.split(self.X):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            model = self.get_model(trial)
            model.fit(X_train, y_train, cat_features=self.cat_features)

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_score = self.metric(y_train, train_pred)
            val_score = self.metric(y_val, val_pred)

            train_scores.append(train_score)
            val_scores.append(val_score)
            diff_scores.append(abs(val_score - train_score))
        mean_val_score = np.mean(val_scores)
        mean_train_score = np.mean(train_scores)
        score_diff = np.mean(diff_scores)
        
        print(f"MEAN Metric on train folds: {mean_train_score:.02f}")
        print(f"MEAN Metric on val folds: {mean_val_score:.02f}")
        print(f"MEAN Metric diff: {score_diff:.02f}")

        if len(self.directions) == 1:
            return mean_val_score
        return mean_val_score, score_diff

    def get_model(self, trial):
        raise NotImplementedError("Subclasses should implement this!")

    def train(self):
        study = optuna.create_study(directions=self.directions)
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True, n_jobs=-1)
        return study

# Classification Tuners
class CatBoostClassifierTuner(BaseModelTuner):
    def get_model(self, trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "depth": trial.suggest_int("depth", 2, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            "od_type": trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
            "od_wait": trial.suggest_int("od_wait", 10, 50),
        }
        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif params["bootstrap_type"] == "Bernoulli":
            params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        return CatBoostClassifier(**params, silent=True)

class LightGBMClassifierTuner(BaseModelTuner):
    def get_model(self, trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 512),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        return lgb.LGBMClassifier(**params)

class XGBoostClassifierTuner(BaseModelTuner):
    def get_model(self, trial):
        params = {
            "verbosity": 0,
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "objective": "binary:logistic",
            "tree_method": trial.suggest_categorical("tree_method", ["exact", "hist", "gpu_hist"]),
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }
        if params["booster"] in ["gbtree", "dart"]:
            params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            params["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)

        return xgb.XGBClassifier(**params)

# Regression Tuners
class CatBoostRegressionTuner(BaseModelTuner):
    def get_model(self, trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "depth": trial.suggest_int("depth", 2, 6),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        }
        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif params["bootstrap_type"] == "Bernoulli":
            params["subsample"] = trial.suggest_float("subsample", 0.4, 1)

        return CatBoostRegressor(**params, silent=True)


    def get_baseline_scores(self):
        kf = KFold(n_splits=4)
        val_scores = []
        train_scores = []

        for train_idx, val_idx in kf.split(self.X):
            X_train, X_val = self.X.loc[train_idx], self.X.loc[val_idx]
            y_train, y_val = self.y.loc[train_idx], self.y.loc[val_idx]
            model = CatBoostRegressor(silent=True)
            model.fit(X_train, y_train, cat_features=self.cat_features)

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_score = self.metric(y_train, train_pred)
            val_score = self.metric(y_val, val_pred)

            train_scores.append(train_score)
            val_scores.append(val_score)

        mean_val_score = np.mean(val_scores)
        mean_train_score = np.mean(train_scores)
        score_diff = abs(mean_val_score - mean_train_score)

        print(f"MEAN Metric on train folds: {mean_train_score:.02f}")
        print(f"MEAN Metric on val folds: {mean_val_score:.02f}")
        print(f"MEAN Metric diff: {mean_val_score - mean_train_score:.02f}")
        return train_scores, val_scores
    
class LightGBMRegressionTuner(BaseModelTuner):
    def get_model(self, trial):
        params = {
            "objective": "regression",
            "metric": "mse",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 512),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }

        return lgb.LGBMRegressor(**params)