# COVID-19 Global Data Analysis and Trend Forecasting

An end-to-end data science portfolio project that analyzes global COVID-19 trends, compares major countries, studies vaccination progress, and forecasts future cases and deaths using machine learning.

## Project Overview

This project is designed for:

- Computer Science student portfolios
- GitHub showcase projects
- Internship applications
- LinkedIn technical posts

It combines data cleaning, exploratory analysis, statistical analysis, forecasting, and dashboard development in one professional repository.

## Objectives

- Load and preprocess a real-world COVID-19 dataset
- Explore global case and death trends over time
- Compare India, the United States, China, Brazil, and the United Kingdom
- Analyze vaccination rollout speed and vaccination coverage
- Study correlations between cases, deaths, and vaccination indicators
- Forecast future case and death trends for 30, 60, and 90 days
- Present the results in both notebook and dashboard format

## Dataset Information

This project uses the public Our World in Data COVID-19 dataset.

- Source page: https://ourworldindata.org/coronavirus
- Direct CSV: `https://covid.ourworldindata.org/data/owid-covid-data.csv`

Why OWID was selected:

- One dataset contains cases, deaths, population, and vaccination indicators
- It is widely cited and easy to reproduce
- It is appropriate for EDA, comparative analytics, and forecasting exercises

> Note: Forecasting requires historical time series data for each country. If a local OWID CSV contains only a snapshot of the latest rows per country, the analysis pipeline will still run and create reports, but forecast outputs may be skipped.

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Jupyter Notebook
- Streamlit

## Project Structure

```text
COVID19-Data-Analysis/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── analysis.py
│   ├── visualization.py
│   └── forecasting.py
├── dashboard/
│   └── app.py
├── reports/
│   ├── module_guide.md
│   ├── github_deployment.md
│   └── linkedin_showcase.md
├── images/
├── scripts/
├── requirements.txt
├── README.md
└── main.py
```

## Installation Steps

```bash
git clone https://github.com/<your-username>/COVID19-Data-Analysis.git
cd COVID19-Data-Analysis
python -m venv .venv
pip install -r requirements.txt
```

Windows activation:

```bash
.venv\Scripts\activate
```

macOS/Linux activation:

```bash
source .venv/bin/activate
```

## Project Workflow

1. Run the main pipeline:

```bash
python main.py
```

2. Open the notebook:

```bash
jupyter notebook
```

3. Launch the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

## What the Project Implements

### 1. Data Collection and Preprocessing

- Dataset loading from a public OWID source
- Missing value handling
- Date conversion
- Duplicate removal
- Feature selection
- Rolling averages and growth-rate transformation
- Cases per million, deaths per million, death rate, vaccination share, and recovery proxy generation

### 2. Exploratory Data Analysis

- Global total cases over time
- Global new cases per day using 7-day averages
- Global total deaths over time
- Global new deaths per day using 7-day averages

### 3. Country-wise Comparisons

Compared countries:

- India
- United States
- China
- Brazil
- United Kingdom

Metrics covered:

- Total cases
- Total deaths
- Death rate
- Recovery trend proxy
- Cases per million
- Vaccination coverage

### 4. Vaccination Analysis

- Total vaccinations
- Fully vaccinated population
- Booster doses
- Vaccination progress curves
- Ranking by vaccination coverage
- Post-vaccination case and death change analysis

### 5. Advanced Statistical Analysis

- Correlation matrix
- Covariance matrix
- Rolling averages
- Growth-rate calculations
- Trend analysis support tables

### 6. Trend Forecasting

Models used:

- Linear Regression
- Polynomial Regression
- Lag-based Random Forest time-series forecasting

Forecast outputs:

- 30-day forecast
- 60-day forecast
- 90-day forecast

Evaluation metrics:

- MAE
- RMSE
- R2 Score

## Results and Business Insights

This project is designed to generate portfolio-ready findings such as:

- Which selected country had the highest total case burden
- Which country had the highest death burden and death rate
- Which country vaccinated the fastest
- Whether case and death averages changed after major vaccination milestones
- How strongly vaccination coverage correlated with mortality and case burden

Public health interpretation focus:

- Population-adjusted metrics matter more than raw totals for fair comparison
- Rolling averages provide more stable signals than raw daily values
- Vaccination analysis should be evaluated together with mortality and case-growth trends

## Real-World Conclusions

- Absolute counts help measure scale, but per-capita metrics reveal intensity
- Vaccination progress can be compared not only by final coverage but also by rollout speed
- Forecasting daily smoothed case and death trends is more practical than forecasting noisy raw spikes
- Decision makers benefit from dashboards that combine descriptive analytics with predictive trends

## Future Improvements

- Add ARIMA or Prophet models for deeper time-series experimentation
- Add continent-wise clustering and anomaly detection
- Integrate hospitalization or testing indicators
- Deploy the dashboard publicly and connect it to scheduled data refresh
- Add unit tests for data preprocessing and forecasting functions

## Screenshots Section

Add screenshots after running the project:

- `images/global_trends.png`
- `images/country_comparison_bars.png`
- `images/vaccination_progress.png`
- forecast evaluation charts generated in `images/`

## Additional Documentation

- Module walkthrough: [reports/module_guide.md](reports/module_guide.md)
- GitHub deployment help: [reports/github_deployment.md](reports/github_deployment.md)
- LinkedIn presentation ideas: [reports/linkedin_showcase.md](reports/linkedin_showcase.md)

## Portfolio Value

This project demonstrates:

- Data preprocessing on a real public dataset
- Strong EDA and storytelling
- Comparative analytics with business insights
- Machine learning based forecasting
- Dashboard development for interactive reporting
- Modular Python project organization suitable for GitHub and interviews
