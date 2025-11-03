import joblib
import os
import numpy as np
from datetime import datetime
from src.mlproject.entity.config_entity import ModelTrainerConfig
from src.mlproject import logging
from box.exceptions import BoxValueError

from catboost import CatBoostRegressor
from sklearn.ensemble import (RandomForestRegressor,
                              AdaBoostRegressor,
                              GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet 
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor

class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.config=config

    def _find_best_models(self, models: dict, params: dict, *, 
                         cv:int = 3, test_size:float = 0.20) -> dict:
        try:
            data_arr = joblib.load(self.config.train_array_path)
            # train_arr, valid_arr = train_test_split(data_arr, test_size=test_size, 
            #                                         random_state=42)

            X_train = data_arr[:, :-1]
            y_train = data_arr[:, -1]
            # X_valid = valid_arr[:, :-1]
            # y_valid = valid_arr[:, -1]

            grid_results = {}
            logging.info("GridSearchCV starts to iterate over each model")
            total_sec = 0
            scoring = {'r2': 'r2', 'neg_mae':'neg_mean_absolute_error'}
            for name, model in models.items():
                start_time = datetime.now()
                param_grid = params.get(name, {})
                gsc = GridSearchCV(
                    estimator = model, 
                    param_grid = param_grid, 
                    cv=cv, 
                    scoring=scoring, 
                    refit='r2', 
                    n_jobs=-1)
                gsc.fit(X_train, y_train)

                best_model = gsc.best_estimator_  # refit model
                train_r2 = gsc.best_score_
                # y_pred = best_model.predict(X_valid)
                # valid_r2 = r2_score(y_valid, y_pred)

                # store all info in one dictionary
                grid_results[name] = {
                    'model': best_model,
                    'best_params': gsc.best_params_,
                    'train_r2': train_r2,
                    # 'valid_r2': valid_r2
                    }

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                total_sec += duration
                dur_min, dur_sec = divmod(int(duration), 60)

                logging.info(f"{name} trained in {dur_min}m {dur_sec}s")
                logging.info(f"train_r2: {train_r2:.4f}")
                logging.info("...............................................")

            total_min, total_sec = divmod(int(total_sec), 60)
            logging.info(f"Total training time: {total_min}m {total_sec}s")
            return grid_results

        except BoxValueError as e:
            logging.error(f"Error: {e}")

    def tune_hyperpara_select_model(self):
        """
        Perform grid search over multiple models using predefined hyperparameter grids
        and store results to artifact and print best model results.

        Parameters: 
            None

        Returns: 
            None
        """
        try:
            logging.info("Starting grid search across all models...")

            models = {
                'ridge': Ridge(),
                'lasso': Lasso(max_iter=50000),
                'enet': ElasticNet(max_iter=30000),
                'rf': RandomForestRegressor(random_state=42),
                'gbr': GradientBoostingRegressor(random_state=42),
                'xgb': XGBRegressor(random_state=42),
                'abr': AdaBoostRegressor(random_state=42),
                'cat': CatBoostRegressor(random_state=42, verbose=False),
                'knn': KNeighborsRegressor()
            }

            params = {
                'ridge': {'alpha': [0.01, 0.1, 1, 10, 100]},
                'lasso': {'alpha': [0.01, 0.1, 1, 10, 100, 200]},
                'enet': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                        'l1_ratio': [0.7, 0.5, 0.3, 0.1]},
                'rf': {'n_estimators': [100, 200, 500],
                    'max_depth': [3, 5, 10, None]
                    },
                'gbr': {'learning_rate': [0.01, 0.05, 0.1],
                        'n_estimators': [100, 200, 500],
                        'subsample': [0.5, 0.7, 0.9, 1.0]},
                'xgb': {'learning_rate': [0.01, 0.05, 0.1],
                        'n_estimators': [100, 200, 500],
                        'max_depth': [3, 5, 7, 10]},
                'abr': {'learning_rate': [0.01, 0.05, 0.1],
                        'n_estimators': [100, 200, 500]},
                'cat': {'learning_rate': [0.01, 0.05, 0.1],
                        'depth': [4, 6, 8, 10],
                        'iterations': [50, 100, 200, 500]},
                'knn': {'n_neighbors': [3, 5, 7, 9, 15, 21]}
            }

            # Perform grid search
            grid_results = self._find_best_models(models=models, params=params, 
                                                 cv=3, test_size=0.25)

            logging.info("Grid search successfully completed!")

            # Save results
            result_path = os.path.join(self.config.root_dir, self.config.grid_results)
            joblib.dump(grid_results, result_path)
            logging.info(f"Grid search results saved to artifacts")

            # Find best model based on R2 score
            best_model = max(grid_results.items(), key=lambda x: x[1]['train_r2'])
            logging.info(f"Best model: {best_model[0]}, based-on R2: {best_model[1]['train_r2']:.3f}")

        except BoxValueError as e:
            logging.error(f"Box configuration error: {e}")
            raise e

        except Exception as e:
            logging.error(f"Unexpected error during grid search: {e}")
            raise e


if __name__ == "__main__":
    print("This module tunes hyperparameters,"
          "picks best models from each estimatore,"
          "saves them to artifacts and isn't meant to be run on its own.")
