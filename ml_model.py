# ml_model.py
"""Machine learning utilities for trade decision making."""

import os
from datetime import datetime, timedelta

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from utils import log


class MLModel:
    def __init__(self, filename='trade_data.csv', model_file='ml_model.pkl'):
        self.filename = filename
        self.model_file = model_file
        self.model = None
        self.last_train_date = None

    def log_trade(self, features: dict, result: bool) -> None:
        """Persist a single trade's features and outcome."""
        features['timestamp'] = datetime.now()
        features['result'] = int(result)
        df = pd.DataFrame([features])
        df.to_csv(self.filename, mode='a', header=not os.path.exists(self.filename), index=False)

    def train_model(self) -> None:
        """Train the RandomForest model using data from the last seven days."""
        log("Treinando modelo de ML com dados dos últimos 7 dias...")
        if not os.path.exists(self.filename):
            log("Nenhum dado disponível para treinar!")
            return

        df = pd.read_csv(self.filename, parse_dates=['timestamp'])
        cutoff = datetime.now() - timedelta(days=7)
        df = df[df.timestamp >= cutoff]

        if len(df) < 50:
            log(f"Dados insuficientes para treinar — apenas {len(df)} trades")
            return

        X = pd.get_dummies(df.drop(columns=['timestamp','result']))
        y = df['result']

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, self.model_file)
        self.model = model
        log("Modelo treinado e salvo!")

    def load_model(self) -> None:
        """Load an existing model or train a new one if absent."""
        if os.path.exists(self.model_file):
            self.model = joblib.load(self.model_file)
            log("Modelo de ML carregado!")
        else:
            log("Modelo inexistente — treinando...")
            self.train_model()

    def predict_high_chance(self, features: dict) -> bool:
        """Return ``True`` if the model predicts probability >= 0.8."""
        if self.model is None:
            self.load_model()

        X = pd.DataFrame([features])
        X = pd.get_dummies(X)

        for col in self.model.feature_names_in_:
            if col not in X.columns:
                X[col] = 0
        X = X[self.model.feature_names_in_]

        proba = self.model.predict_proba(X)[0][1]
        return proba >= 0.8

    def check_and_train_daily(self):
        """Train the model at 6 AM once per day."""
        now = datetime.now()
        if (
            now.hour == 6
            and (
                self.last_train_date is None
                or self.last_train_date.date() < now.date()
            )
        ):
            self.train_model()
            self.last_train_date = now