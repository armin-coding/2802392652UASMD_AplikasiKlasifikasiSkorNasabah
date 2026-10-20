import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# class untuk membersihkan data, memisahkan train/test, dan membangun pipeline transformasi
class DataPreprocessor:
    def __init__(self, target_column: str, test_size: float = 0.2, random_state: int = 42):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None
        self.num_cols = []
        self.cat_cols = []

    @staticmethod
    def _clean_numeric(series: pd.Series) -> pd.Series:
        cleaned = series.astype(str).str.replace(r'[^0-9.]', '', regex=True)
        return pd.to_numeric(cleaned, errors='coerce')

    @staticmethod
    def _parse_credit_age(s) -> float:
        if pd.isna(s):
            return np.nan
        years = re.search(r'(\d+)\s+Year', str(s))
        months = re.search(r'(\d+)\s+Month', str(s))
        y = int(years.group(1)) if years else 0
        m = int(months.group(1)) if months else 0
        return y * 12 + m

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        target_drops = [
            'unnamed: 0', 'id', 'customer_id', 'name',
            'ssn', 'month', 'type_of_loan', 'backlogs',
            'total_emi_per_month', 'monthly_inhand_salary',
            'credit_utilization_ratio', 'payment_of_min_amount',
            'num_credit_card', 'num_credit_inquiries',
            'delay_from_due_date', 'monthly_balance',
            'interest_rate', 'num_bank_accounts'
        ]

        df = df.drop(columns=[col for col in df.columns if col.lower() in target_drops])

        str_num_cols = [
            'Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
            'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly'
        ]
        for col in str_num_cols:
            if col in df.columns:
                df[col] = self._clean_numeric(df[col])

        if 'Credit_History_Age' in df.columns:
            df['Credit_History_Age'] = df['Credit_History_Age'].apply(self._parse_credit_age)

        if 'Age' in df.columns:
            df.loc[(df['Age'] < 18) | (df['Age'] > 100), 'Age'] = np.nan
        if 'Num_of_Loan' in df.columns:
            df.loc[df['Num_of_Loan'] < 0, 'Num_of_Loan'] = np.nan

        if 'Occupation' in df.columns:
            df['Occupation'] = df['Occupation'].replace('_______', np.nan)
        if 'Credit_Mix' in df.columns:
            df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)
        if 'Payment_Behaviour' in df.columns:
            df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', np.nan)

        print("[CLEANING] Pembersihan kotor struktural berhasil.")
        return df

    def split_data(self, df: pd.DataFrame):
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        self.num_cols = list(X.select_dtypes(include=['int64', 'float64']).columns)
        self.cat_cols = list(X.select_dtypes(include=['object', 'category']).columns)

        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_cols),
                ('cat', categorical_transformer, self.cat_cols)
            ]
        )

        X_train_trans = self.preprocessor.fit_transform(X_train)
        joblib.dump(self.preprocessor, 'models/preprocessor.pkl', compress=3)

        print(f"[INFO] Preprocessor siap. (Memproses {len(self.num_cols)} numerik & {len(self.cat_cols)} kategorikal)")
        return X_train_trans

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            raise ValueError("[ERROR] Preprocessor belum dilatih melalui fit_transform!")
        return self.preprocessor.transform(X_test)