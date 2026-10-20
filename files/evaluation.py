import json
import pickle
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, classification_report

# class untuk mengekspor metrik serta metadata untuk kebutuhan produksi
class ModelEvaluator:
    def __init__(self, model, model_name: str, X_test, y_test, num_cols: list, cat_cols: list):
        self.model      = model
        self.model_name = model_name
        self.X_test     = X_test
        self.y_test     = y_test
        self.num_cols   = num_cols
        self.cat_cols   = cat_cols
        self.y_pred     = self.model.predict(X_test)

    def evaluate(self) -> dict:
        # menghitung metrik utama, mengekspor ke json, dan log ke mlflow
        macro_f1 = f1_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred, output_dict=True, zero_division=0)

        print("\n[EVALUASI FINAL]")
        print(f"Nama Model Terpilih : {self.model_name}")
        print(f"Akurasi             : {accuracy:.4f}")
        print(f"Macro F1-Score      : {macro_f1:.4f}")

        metrics_summary = {
            'model_name': self.model_name,
            'accuracy': accuracy,
            'macro_f1_score': macro_f1,
            'detailed_report': report
        }

        # ekspor metrik ke json
        with open('metrics/evaluation_report.json', 'w') as f:
            json.dump(metrics_summary, f, indent=4)
        print("[INFO] Laporan performa diekspor ke 'metrics/evaluation_report.json'.")

        # log ke mlflow
        with mlflow.start_run(run_name=f"Final_Evaluation_{self.model_name}", nested=True):
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_macro_f1", macro_f1)
            mlflow.sklearn.log_model(self.model, "best_model_deployment")

        return metrics_summary

    def save_artifacts(self):
        # menyimpan metadata kolom.
        metadata = {
            'model_name': self.model_name,
            'numerical_columns': self.num_cols,
            'categorical_columns': self.cat_cols
        }
        
        with open('models/model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        print("[INFO] File 'model_metadata.pkl' berhasil diamankan di folder 'models/'.")