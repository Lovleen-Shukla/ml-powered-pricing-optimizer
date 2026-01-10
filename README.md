## Exploratory Data Analysis (EDA)

- Cleaned transactional data to remove returns and invalid prices
- Aggregated product-level daily demand
- Validated inverse price-demand relationship
- Identified seasonal patterns
- Prepared data for elasticity modeling and demand forecasting

## Demand Modeling

- Built demand forecasting models using price and time-based features
- Compared interpretable baseline (Linear Regression) with non-linear model (Random Forest)
- Random Forest captured complex demand patterns and outperformed baseline
- Feature importance confirmed price sensitivity and seasonality effects

## Pricing Optimization

- Used trained demand model to simulate demand across multiple price points
- Estimated expected revenue for each price
- Identified optimal price that maximizes revenue
- Visualized revenue-price curve to support pricing decisions


## Dashboard & Monitoring

- Built an interactive Streamlit dashboard for pricing optimization
- Enabled product-level price simulation and revenue visualization
- Displayed optimal price recommendations with predicted demand
- Designed system for real-world ML decision support
