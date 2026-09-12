# F1 Lap Time Predictor

## About

F1 Lap Time Predictor is a Python-based machine learning project designed to forecast Formula 1 lap times by separating intra-stint tire degradation from overall race fuel burn. Using historical F1 timing and pit stop data (e.g., from the 2013 Spanish Grand Prix), the pipeline cleans pit transition noise and statistical pace anomalies, computes stint boundaries and tire age, and trains degradation-aware predictive models to evaluate late-race stint pace without temporal data leakage.

## Tech Stack

- Language: Python
- Data Analysis: numpy, pandas
- Machine Learning: scikit-learn
- Visualization: matplotlib
- Environment: Jupyter Notebook

---

## Project Structure

```text
f1_lap_time_predictor/
├── data
├── f1_lap_time_predictor.ipynb
├── lap_time_comparison.png
├── README.md
└── requirements.txt
```

---

## How to Run the Project

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook

```bash
jupyter notebook f1_lap_time_predictor.ipynb
```

---

## Features Implemented

- Dynamic race and top-finisher driver selection 
- Pit transition filtering (removal of pit in-laps and out-laps)
- Outlier filtering using statistical thresholding (1.5× median pace / 3-sigma rules)
- Feature engineering for `stint_number` and `tire_age` based on pit entry boundaries
- Temporal stint-based train/test splitting (training on early stints 1 to N-1, testing on final stint N)
- Baseline model implementation (`LinearRegression` using grid and lap number)
- Enhanced model implementation (`Ridge` regression using grid, lap, stint number, and tire age)
- Comprehensive metric evaluation logging (RMSE and MAE in seconds)
- Automated plot generation comparing actual vs. predicted final stint lap times (`lap_time_comparison.png`)

---

## Additional Features Added

- Dynamic target driver selection highlighting the longest final stint for visual plotting.
- Detailed stdout and markdown logging summarizing data cleaning drop counts and train/test split metrics.

---

## Concepts Learned While Completing the Task

- Handling domain-specific telemetry anomalies (pit transitions vs. true racing pace)
- Preventing temporal data leakage in chronological race scenarios
- Isolating opposing feature effects (fuel mass reduction vs. tire compound wear)
- Regularized linear regression modeling using `scikit-learn`
- Structuring modular machine learning pipelines in Jupyter Notebooks
- Evaluating model error metrics (RMSE/MAE) in domain-specific units (seconds)
