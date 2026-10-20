import os
import mlflow
from data_ingestion import DataIngestion
from preprocessing import DataPreprocessor
from train import ModelTrainer
from evaluation import ModelEvaluator

def run_pipeline():
    
    print("MEMULAI PIPELINE (DATA INGESTION -> EVALUATION)")


    # SETUP MLFLOW
    # Inisialisasi tracking URI dan pastikan nama eksperimen konsisten
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("2802392652_DanielSebastianWinata_MDPROJECT_UAS")

    # buka parent run untuk merangkum seluruh proses pipeline
    with mlflow.start_run(run_name="Production_Pipeline_Run"):
        

        # TAHAP 1: DATA INGESTION

        print("\n>>> TAHAP 1: DATA INGESTION")
        # pastikan path file CSV ADA di data
        ingestion = DataIngestion('data/data_C.csv') 
        df = ingestion.load_data()


        # TAHAP 2: DATA PREPROCESSING

        print("\n>>> TAHAP 2: DATA PREPROCESSING")
        preprocessor = DataPreprocessor(target_column='Credit_Score')
        
        # bersiihn noise dan kolom yang tidak relevan 
        df_clean = preprocessor.clean_data(df)
        
        # Split data sebelum imputasi untuk mencegah data leakage
        X_train, X_test, y_train, y_test = preprocessor.split_data(df_clean)
        
        # bangun pipeline transformasi dan simpan preprocessor.pkl
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans  = preprocessor.transform(X_test)
        
        # ambil metadata kolom untuk keperluan Streamlit nanti
        num_cols = preprocessor.num_cols
        cat_cols = preprocessor.cat_cols


        # TAHAP 3: MODEL TRAINING & TUNING

        print("\n>>> TAHAP 3: MODEL TRAINING & SELEKSI DINAMIS")
        trainer = ModelTrainer(X_train_trans, y_train, X_test_trans, y_test)
        
        # 5 model baseline ditrain
        trainer.run_baseline()
        
        # hyperparameter tuning, bandingkan dengan baseline
        # pilih yang terbaik (Macro F1 tertinggi), dan simpan best_model.pkl
        best_model, best_name, all_results = trainer.train_and_tune()


        # TAHAP 4: MODEL EVALUATION

        print("\n>>> TAHAP 4: MODEL EVALUATION & ARTIFACT EXPORT")
        evaluator = ModelEvaluator(
            model=best_model,
            model_name=best_name,
            X_test=X_test_trans,
            y_test=y_test,
            num_cols=num_cols,
            cat_cols=cat_cols
        )
        
        # hitung metrik akhir, ekspor ke JSON, dan log model ke MLflow
        evaluator.evaluate()
        
        # simpan metadata model (model_metadata.pkl) untuk blueprint Streamlit
        evaluator.save_artifacts()


    print("PIPELINE SELESAI DIEKSEKUSI DENGAN SUKSES!")
    print("Artefak preprocessor.pkl, best_model.pkl, dan model_metadata.pkl")
    print("sudah tersedia di folder 'models/'.")
    print("python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1 || di terminal jika mau liat dashboard eksperimennya")


if __name__ == "__main__":
    run_pipeline()