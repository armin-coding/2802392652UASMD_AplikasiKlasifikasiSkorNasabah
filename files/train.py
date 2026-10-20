import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# class untuk melatih baseline model, melakukan tuning, 
# dan secara dinamis memilih serta menyimpan model dengan evaluasi terbaik
class ModelTrainer:
    BASELINE_MODELS = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Decision Tree':       DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=11),
        'Random Forest':       RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    def __init__(self, X_train, y_train, X_test, y_test, random_state: int = 42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
        self.random_state = random_state
        
        self.baseline_results = []
        self.tuned_results    = []
        self.trained_models   = {} 
        
        self.best_model = None
        self.best_name  = None

    def _score(self, name: str, y_pred) -> dict:
        return {
            'Model':           name,
            'Accuracy':        accuracy_score(self.y_test, y_pred),
            'Macro Precision': precision_score(self.y_test, y_pred, average='macro', zero_division=0),
            'Macro Recall':    recall_score(self.y_test, y_pred, average='macro', zero_division=0),
            'Macro F1':        f1_score(self.y_test, y_pred, average='macro', zero_division=0),
        }

    def run_baseline(self) -> pd.DataFrame:
        print("[INFO] Memulai training model baseline...")
        self.baseline_results = []
        
        for name, model in self.BASELINE_MODELS.items():
            print(f'Training {name}...', end=' ')
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model 
            
            y_pred = model.predict(self.X_test)
            row = self._score(name, y_pred)
            self.baseline_results.append(row)
            print(f"Accuracy={row['Accuracy']:.4f}, Macro F1={row['Macro F1']:.4f}")

        return pd.DataFrame(self.baseline_results).sort_values('Macro F1', ascending=False).reset_index(drop=True)

    def train_and_tune(self) -> tuple:
        print('\n[INFO] Memulai Hyperparameter Tuning...')
        
        #  tuning random forest 
        rf_param_grid = {
            'n_estimators':      [100, 200],
            'max_depth':         [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        rf_grid = GridSearchCV(
            RandomForestClassifier(class_weight='balanced', random_state=self.random_state, n_jobs=-1),
            rf_param_grid, cv=3, scoring='f1_macro', n_jobs=-1
        )

        with mlflow.start_run(run_name="Tuning_RandomForest", nested=True):
            rf_grid.fit(self.X_train, self.y_train)
            rf_pred = rf_grid.best_estimator_.predict(self.X_test)
            rf_f1 = f1_score(self.y_test, rf_pred, average='macro')
            mlflow.log_params(rf_grid.best_params_)
            mlflow.log_metric("val_macro_f1", rf_f1)
            mlflow.sklearn.log_model(rf_grid.best_estimator_, "model_RandomForest")

        print(f'Best RF params  : {rf_grid.best_params_}')
        
        #  tuning gradient boosting 
        gb_param_grid = {
            'n_estimators':  [100, 200],
            'max_depth':     [3, 5],
            'learning_rate': [0.05, 0.1, 0.2]
        }
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=self.random_state),
            gb_param_grid, cv=3, scoring='f1_macro', n_jobs=-1
        )

        with mlflow.start_run(run_name="Tuning_GradientBoosting", nested=True):
            gb_grid.fit(self.X_train, self.y_train)
            gb_pred = gb_grid.best_estimator_.predict(self.X_test)
            gb_f1 = f1_score(self.y_test, gb_pred, average='macro')
            mlflow.log_params(gb_grid.best_params_)
            mlflow.log_metric("val_macro_f1", gb_f1)
            mlflow.sklearn.log_model(gb_grid.best_estimator_, "model_GradientBoosting")

        print(f'Best GB params  : {gb_grid.best_params_}')

        #  evaluasi hasil tuning 
        tuned_models = {
            'Random Forest (Tuned)':     rf_grid.best_estimator_,
            'Gradient Boosting (Tuned)': gb_grid.best_estimator_
        }
        
        self.trained_models.update(tuned_models)
        self.tuned_results = []
        for name, model in tuned_models.items():
            y_pred = model.predict(self.X_test)
            self.tuned_results.append(self._score(name, y_pred))

        #  seleksi final otomatis 
        all_results = pd.DataFrame(self.baseline_results + self.tuned_results)\
                        .sort_values('Macro F1', ascending=False).reset_index(drop=True)
        
        self.best_name = all_results.iloc[0]['Model']
        self.best_model = self.trained_models[self.best_name]

        print(f'\n====================================')
        print(f'MODEL TERBAIK: {self.best_name}')
        print(f'MACRO F1     : {all_results.iloc[0]["Macro F1"]:.4f}')
        print(f'====================================')

        joblib.dump(self.best_model, 'models/best_model.pkl', compress=('lzma', 9))
        print(f"[INFO] Artefak '{self.best_name}' berhasil disimpan ke 'models/best_model.pkl'")

        return self.best_model, self.best_name, all_results