# KhetReady AI
AI-based prediction of agricultural field operability and operation delay risk.

## Pipeline
Data -> cleaning -> temporal features -> agricultural operation context -> proxy labels -> ML -> evaluation -> explainability -> Streamlit.

### Important methodological note
The starter project creates transparent **proxy operational-risk labels** from rainfall and soil-moisture conditions. These are NOT observed farmer delay records. This lets the software pipeline run end-to-end while keeping the limitation explicit. For the final research version, replace the proxy-label function with real historical operation-outcome labels if such records can be obtained.

## Official data sources
- Government of India OGD soil moisture: https://data.gov.in/catalog/daily-data-soil-moisture
- Government of India OGD rainfall: https://data.gov.in/catalog/rainfall-india

## Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/make_demo_data.py
python src/prepare_data.py
python src/train.py
python src/evaluate.py
streamlit run app/streamlit_app.py
```

The demo dataset is only for testing the pipeline. For the final project, replace it with the selected official data and document the exact resource used.
