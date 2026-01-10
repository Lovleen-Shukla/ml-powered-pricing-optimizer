import joblib
from sklearn.ensemble import RandomForestRegressor

def train_demand_model(X, y):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model
