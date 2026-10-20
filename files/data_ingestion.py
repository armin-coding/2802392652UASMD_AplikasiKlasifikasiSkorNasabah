import os
import pandas as pd

# class untuk menangani pemuatan dataset mentah dan penyiapan direktori proyek.
class DataIngestion:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self._create_directories()

    def _create_directories(self):
        directories = ['data', 'models', 'metrics', 'logs']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"[INFO] Folder '{directory}' siap digunakan.")

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"[ERROR] File data tidak ditemukan di path: {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        print(f"[INFO] Data berhasil dimuat. Ukuran dataset: {self.df.shape}")
        return self.df