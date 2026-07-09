# F1 World Championship Analyzer & Predictor

A data analysis project exploring Formula 1 World Championship history — reconstructing official driver and constructor standings from raw race data, visualizing title battles across seasons, and (eventually) predicting championship outcomes.

Built as part of a Python training project (Udemy 100 Days of Code) to apply pandas, data cleaning, and eventually scikit-learn to a real, messy, historical dataset.

## Project Status
🚧 In progress — currently at the data cleaning and reconstruction stage.

## What's been done so far
- Loaded and explored raw F1 data (races, results, drivers, constructors, standings) from the [Kaggle F1 dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
- Identified and handled a real data quality issue: missing finishing positions were stored as the string `\N` (from the original MySQL export) rather than proper null values
- Built a reusable `get_driver_standings(year)` function that reconstructs a season's final driver standings from raw race + sprint race results
- Built an equivalent `get_constructor_standings(year)` function for team standings
- Validated both functions against the official standings data — confirmed exact point-for-point matches, including correctly accounting for 2021-onward sprint race points

## Coming up
- Visualizations of championship battles (points progression by race, era comparisons)
- A predictive model estimating championship outcomes from partial-season data

## Setup
This project uses a Python virtual environment. To run it locally:

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
pip install pandas jupyter matplotlib
```

Raw data (not included in this repo) can be downloaded from the [Kaggle dataset link above](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) and placed in `data/raw/`.

## Author
Bhavik Rustagi