import pandas as pd
import joblib
import pickle

# class untuk menangani proses inferensi dari input json hingga prediksi akhir
class ModelInferencer:
    def __init__(self, model_dir: str = 'models'):
        # muat semua artefak produksi
        self.model = joblib.load(f'{model_dir}/best_model.pkl')
        self.preprocessor = joblib.load(f'{model_dir}/preprocessor.pkl')
        
        # muat blueprint kolom
        with open(f'{model_dir}/model_metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
            
        self.num_cols = self.metadata['numerical_columns']
        self.cat_cols = self.metadata['categorical_columns']
        
        # gabungkan untuk mengetahui ekspektasi urutan fitur
        self.expected_cols = self.num_cols + self.cat_cols

    def predict_from_json(self, json_data: dict) -> str:
        # konversi dictionary/json ke dataframe pandas (1 baris)
        df_input = pd.DataFrame([json_data])
        
        # pastikan semua kolom yang dibutuhkan ada, jika tidak isi dengan none/nan
        for col in self.expected_cols:
            if col not in df_input.columns:
                df_input[col] = None
                
        # urutkan kolom secara ketat sesuai dengan urutan saat training
        df_input = df_input[self.expected_cols]
        
        # jalankan transformasi data
        X_trans = self.preprocessor.transform(df_input)
        
        # lakukan prediksi
        prediction = self.model.predict(X_trans)
        
        # kembalikan hasil prediksi pertama (karena inputnya cuma 1 baris)
        return prediction[0]


# blok pengujian lokal (hanya jalan kalau file ini dieksekusi langsung)
if __name__ == "__main__":
    # contoh payload json dari streamlit nanti
    dummy_json = {
        "Age": "25",
        "Annual_Income": "50000",
        "Outstanding_Debt": "1500.50",
        "Credit_Mix": "Good",
        "Payment_Behaviour": "High_spent_Medium_value_payments",
        "Credit_History_Age": "5 Years and 2 Months"
        # fitur lain bisa ditambahkan di sini untuk ngetes
    }
    
    print("[INFO] Menguji inferencing.py secara lokal...")
    try:
        inferencer = ModelInferencer()
        hasil = inferencer.predict_from_json(dummy_json)
        print(f"Prediksi berhasil! Hasil: {hasil}")
    except Exception as e:
        print(f"Terjadi error saat prediksi: {e}")

        

# File inferencing.py ini bertugas sebagai "otak" di belakang antarmuka Streamlit. 
# Tugasnya adalah menerima input dari pengguna (dalam bentuk JSON atau dictionary), 
# menyusunnya agar sesuai dengan struktur saat training (menggunakan bantuan metadata), melakukan transformasi, 
# dan mengeluarkan hasil prediksi