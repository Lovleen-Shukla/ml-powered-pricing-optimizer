import numpy as np
import pandas as pd

def find_optimal_price(
    model,
    base_row,
    features,
    price_min_factor=0.7,
    price_max_factor=1.3,
    steps=20
):
    current_price = base_row["Price"]

    price_range = np.linspace(
        current_price * price_min_factor,
        current_price * price_max_factor,
        steps
    )

    results = []

    for price in price_range:
        row = base_row.copy()
        row["Price"] = price

        X = pd.DataFrame([row[features]])
        demand = model.predict(X)[0]
        revenue = price * max(demand, 0)

        results.append({
            "price": price,
            "predicted_demand": demand,
            "expected_revenue": revenue
        })

    return pd.DataFrame(results)
