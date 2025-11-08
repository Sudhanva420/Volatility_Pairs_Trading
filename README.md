# Volatility Pairs Trading

This repository presents an analysis of **implied volatility relationships** between **Bank Nifty** and **Nifty** indices.  
Using a **minute-level options dataset** containing implied volatilities and time-to-expiry values, we explore their correlation structure to design a **medium-frequency pairs trading strategy** that profits from deviations in their volatility spread.

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
