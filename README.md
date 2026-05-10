# Deep Learning and Its Application to Stock Markets

This project explores application of Deep Learning into financial datasets, including stock price prediction, trading signal identification, risk/return management, and portfolio construction. 

---
## Project Structure
```
project-root/
├── api/
│   ├── app.py                          # FastAPI REST API server
│   └── test_load_model.py              # Model loading smoke test
├── data/
│   ├── hose                            # Contain historical prices and financial information of 12 tickers in Vietnam market
│   └── nasdaq                          # Contain historical prices of 11 tickers in nasdaq market
├── models/
│   └── task2_3_vietnam_model.keras     # Trained CNN price forecast model
├── notebooks/
│   └── project-notebook.ipynb          # Project notebook structured as Task 1-4
├── pipeline/
│   └── project-notebook.ipynb          # AI workflow diagram 
├── ui/
│   └── streamlit_app.py                # Streamlit web interface
└── requirements.txt                    # Contain required packages to run full stack  
```
---
## Tasks Overview as Structured in the Notebook 
 
| Task | Description |
|---|---|
| 1.1 | Nasdaq multi-feature price prediction |
| 1.2 | Nasdaq k-th day forecast |
| 1.3 | Nasdaq k consecutive days forecast |
| 2.1 | Vietnam multi-feature price prediction |
| 2.2 | Vietnam k-th day forecast |
| 2.3 | Vietnam k consecutive days forecast |
| 3.1 | Buy signal identification (HPG) |
| 3.2 | Sell signal identification (HPG) |
| 4.1 | Profitable stock selection |
| 4.2 | Risk management and stock exclusion |
| 4.3 | Portfolio optimisation (Markowitz) |
| 5.1 | Model deployment as REST API (FastAPI) |
| 5.2 | Web interface (Streamlit) |
| 5.3 | AI engineering workflow design |
 
---
## Setup
 
### Requirements
 
- Python 3.10+
- pip
### Installation
This provides suggested setup in VS code for Window. The exact steps may vary across devices and need customization.
Open project root folder in VS code, then run on the new terminal:
```
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate
 
# Install dependencies
pip install -r requirements.txt
```
---
## Data
 
The project uses two datasets to train all models in the notebook. Task 1 uses nasdaq dataset, and task 2, 3 and 4 use hose dataset. 
 
**Nasdaq dataset**
- Format: one CSV per ticker, columns - `Date, Low, Open, Volume, High, Close, Adjusted Close`
- Place files in: `data/nasdaq/`
**Vietnam HOSE dataset**
- Format: one CSV per ticker, named `{TICKER}-VNINDEX-History.csv`
- Additional files: `{TICKER}-VNINDEX-Dividend.csv`, `{TICKER}-VNINDEX-Industry.csv`, `{TICKER}-VNINDEX-Finance.csv`,  `ticker-overview.csv`
- Place files in: `data/hose/`
In the notebook, update the `NASDAQ_DATA_DIR` and `HOSE_DATA_DIR` variables at the top of each notebook to point to your local data folders.
 
---
## Notebook

The notebook has file extension `.ipynb` as Jupyter Python notebook. 
It includes all codes from data loading, preprocessing, model training and evaluation.

---
## Running the API
 
```
cd api
python -m uvicorn app:app
```
 
The API will be available at:
- **URL:** `http://127.0.0.1:8000`
- **Interactive docs:** `http://127.0.0.1:8000/docs`
- **Health check:** `http://127.0.0.1:8000/health`

To submit a price prediction request directly to the API, post a JSON body containing
a ticker string and an instances array of 30 rows, each with five values in the order
`[Open, High, Low, Close, Volume]`. Below is an example:
 
```
{
  "ticker": "HPG",
  "instances": [
    [21000, 21500, 20800, 21200, 500000],
    [21200, 21700, 21000, 21400, 480000],
    [21400, 21900, 21200, 21600, 520000],
    [21600, 22000, 21400, 21800, 510000],
    [21800, 22200, 21600, 22000, 490000],
    [22000, 22400, 21800, 22100, 505000],
    [22100, 22500, 21900, 22300, 495000],
    [22300, 22700, 22100, 22500, 515000],
    [22500, 22900, 22300, 22700, 525000],
    [22700, 23100, 22500, 22900, 535000],
    [22900, 23300, 22700, 23100, 545000],
    [23100, 23500, 22900, 23300, 555000],
    [23300, 23700, 23100, 23500, 565000],
    [23500, 23900, 23300, 23700, 575000],
    [23700, 24100, 23500, 23900, 585000],
    [23900, 24300, 23700, 24100, 595000],
    [24100, 24500, 23900, 24300, 605000],
    [24300, 24700, 24100, 24500, 615000],
    [24500, 24900, 24300, 24700, 625000],
    [24700, 25100, 24500, 24900, 635000],
    [24900, 25300, 24700, 25100, 645000],
    [25100, 25500, 24900, 25300, 655000],
    [25300, 25700, 25100, 25500, 665000],
    [25500, 25900, 25300, 25700, 675000],
    [25700, 26100, 25500, 25900, 685000],
    [25900, 26300, 25700, 26100, 695000],
    [26100, 26500, 25900, 26300, 705000],
    [26300, 26700, 26100, 26500, 715000],
    [26500, 26900, 26300, 26700, 725000],
    [26700, 27100, 26500, 26900, 735000]
  ]
}
```
Note that the realistic financial data should be used to ouput meaningful information.

**Sample response**:
 
```
{
  "ticker": "HPG",
  "prediction": [21480.5, 21392.3, 21310.7, 21250.1, 21190.4, 21140.8, 21100.2],
  "horizon": 7
}
```
 
---
 
## Running the Web Interface
 
Make sure the API is running first, then open a second terminal:
 
```
cd ui
streamlit run streamlit_app.py
```

The interface opens automatically at `http://localhost:8501`.
 
**How to use:**
1. Select input mode, either *Close prices only* or *Full OHLCV*
2. Fill in the 30-row table with recent trading data. Note that the realistic financial data should be inputted to ouput meaningful information.
3. Click **Predict**
4. View the 7-day forecast chart, metric cards, and detailed table
---

## AI Engineering Workflow
 
The automated pipeline is designed with the following tools. Please visit in the `pipeline` foler for full workflow diagram.
 
| Tool | Role |
|---|---|
| **Airbyte** | Ingest raw HOSE stock data from CSV or live market API into PostgreSQL |
| **dbt** | Transform raw prices into feature tables (SMA, RSI, log returns) |
| **Airflow** | Orchestrate the daily pipeline: ingest -> transform -> predict -> store |
| **PostgreSQL** | Persist raw data, transformed features, and model predictions |
| **FastAPI** | Serve pre-computed predictions via REST API |
| **Streamlit** | Display forecasts in the web dashboard |
 
The Airflow DAG runs daily at 18:00 on weekdays, after market close.
 
---

## Limitations
 
- Buy/sell signal thresholds are calibrated and optimized for HPG only
- Deployment was only on price forecasting model
- Cross-validation produces two folds due to dataset size constraints
- The historical prices were only updated to 2022. It implicitly assumes that the historical return distribution of the 2019-2022 test period is representative of the near future.
---

## References
- Airbyte, Inc. (2024). Airbyte (Version 0.50.0) [Computer software]. https://airbyte.com/
- Apache Software Foundation. (2024). Apache Airflow (Version 2.8.1) [Computer software]. https://airflow.apache.org/
- dbt Labs, Inc. (2024). dbt (Version 1.7) [Computer software]. https://www.getdbt.com/
- Deep, A., Monico, C., Shirvani, A., Rachev, S., & Fabozzi, F. (2024). Assessing the impact of technical indicators on machine learning models for stock price prediction. 10.48550/arXiv.2412.15448.15
- Kundu, A., & Tiwari, S. (2026). Stock market trend analysis using data science techniques. 10.5281/zenodo.20029312.
- Liu, T. (2026). A comparative study of transformer-based and classical models for financial time-series forecasting. Journal of Risk and Financial Management,
19(3), Article 203. https://doi.org/10.3390/jrfm19030203
- MongoDB, Inc. (2024). MongoDB (Version 7.0) [Computer software]. https://www.mongodb.com/
- Nabipour, M., Nayyeri, P., Jabani, H., Band, S., & Mosavi, A. (2020). Predicting stock market trends using machine learning and deep learning algorithms via continuous and binary data; a comparative analysis on the Tehran stock exchange. IEEE Access. PP. 1-1. 10.1109/ACCESS.2020.3015966.
- The PostgreSQL Global Development Group. (2024). PostgreSQL (Version 16.2) [Computer software]. https://www.postgresql.org/
