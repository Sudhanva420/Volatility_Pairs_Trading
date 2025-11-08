# Volatility Pairs Trading

This repository contains an analysis of implied volatility relationships between Bank Nifty and Nifty indices. The dataset used is a minute-level Options dataset which contains the Implied Volatilities of the 2 instruments along with their time to expiry. We take advantage of their correlation to build a medium frequency pair-trading strategy to profit off the spread between them.

---

## Project Structure

- **`Main.ipynb`** – Contains the complete workflow:
  - Data preprocessing and feature engineering  
  - Exploratory data analysis and visualization  
  - Statistical modeling and hypothesis testing  
  - Backtesting and evaluation of trading strategies  

- **`strategy_base.py`** – Implements the baseline **z-score mean reversion** strategy.  
- **`strategy_strict.py`** – Extends the base model with **stricter entry/exit thresholds** and improved risk management.  
- **`strategy_ml.py`** – Machine learning–based model that captures nonlinear dependencies to outperform the base strategy.

## Dataset

- **Frequency:** 1-minute  
- **Fields:** Implied volatility (Bank Nifty & Nifty), time to expiry, timestamp
  

All preprocessing, feature creation, and data exploration steps are performed within the `Main.ipynb` notebook.  
The trading strategy logic is modularized into separate `.py` files.
