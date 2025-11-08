# Volatility_Pairs_Trading

This repository contains an analysis of implied volatility relationships between Bank Nifty and Nifty indices. The dataset used is a minute-level Options dataset which contains the Implied Volatilities of the 2 instruments along with their time to expiry. We take advantage of their correlation to build a medium frequency pair-trading strategy to profit off the spread between them.

One notebook(main.ipynb) contains all the codes implemented along with the statistical modelling aspect before implementing any strategies and explanation of each part, and 3 separate .py files with each function's code for each strategy's logic

Models used for trading strategy-

Base Model is a z-score based mean reversion strategy 
The second model is built using a Machine learning approach to outperform the Base Model

