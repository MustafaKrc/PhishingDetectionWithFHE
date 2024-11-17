import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, vectorizer_type: str = 'tfidf', max_features: int = 5000):
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        elif vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        else:
            raise ValueError("Unsupported vectorizer type.")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scale_factor = 2 ** 7 - 1  # For 7-bit quantization

    def load_data(self, file_path: str) -> pd.DataFrame:
        data = pd.read_csv(file_path)
        data.dropna(inplace=True)
        data['phishing'] = data['phishing'].astype(int)
        return data

    def split_data(self, data: pd.DataFrame, test_size: float = 0.2 , random_state: int = 42):
        X = data['text']
        y = data['phishing']
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    def fit_transform(self, X_train_text: pd.Series):
        X_train_vectors = self.vectorizer.fit_transform(X_train_text)
        X_train_scaled = self.scaler.fit_transform(X_train_vectors.toarray())
        X_train_quantized = (X_train_scaled * self.scale_factor).astype('float32')
        return X_train_quantized

    def transform(self, X_text: pd.Series):
        X_vectors = self.vectorizer.transform(X_text)
        X_scaled = self.scaler.transform(X_vectors.toarray())
        X_quantized = (X_scaled * self.scale_factor).astype('float32')
        return X_quantized
