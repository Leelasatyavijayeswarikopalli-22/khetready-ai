import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import joblib, pandas as pd, streamlit as st
from src.feature_engineering import add_environment_features

MODEL=Path("models/random_forest.joblib")
st.set_page_config(page_title="KhetReady AI",page_icon="🌾",layout="wide")
st.title("🌾 KhetReady AI")
st.subheader("Agricultural Field Operability & Operation Delay Risk")
st.info("Prototype: risk labels are transparent proxy labels, not observed farmer-delay records.")

if not MODEL.exists():
    st.error("Run the data and training commands first.")
    st.stop()

with st.sidebar:
    rain=st.number_input("Rainfall today (mm)",0.0,300.0,2.0)
    moisture=st.number_input("Soil moisture",0.0,100.0,25.0)
    operation=st.selectbox("Operation",["sowing","tillage","fertilization","spraying","harvesting"])
    crop=st.selectbox("Crop",["rice","wheat","maize","pulses","vegetables"])
    stage=st.selectbox("Crop stage",["pre_sowing","vegetative","flowering","maturity","harvest"])
    area=st.number_input("Field area (acres)",0.1,500.0,5.0)

# For the interactive prototype, build a short synthetic history around the entered values.
# Production use should compute rolling features from actual historical observations.
today=pd.Timestamp.today().normalize()
hist=pd.DataFrame([
    {"date":today-pd.Timedelta(days=i),"district":"Interactive",
     "rainfall_mm":rain*(1 if i==0 else 0.5),"soil_moisture":moisture}
    for i in range(14,-1,-1)
])
row=add_environment_features(hist).tail(1).copy()
row["operation_type"]=operation
row["crop"]=crop
row["crop_stage"]=stage
row["field_area_acres"]=area
row["operation_duration_hours"]={"sowing":5,"tillage":6,"fertilization":3,"spraying":2,"harvesting":8}[operation]

drop=["date","district","risk_label","risk_label_name","risk_score_proxy","recovery_class"]
features=[c for c in row.columns if c not in drop]
model=joblib.load(MODEL)
pred=int(model.predict(row[features])[0])
proba=model.predict_proba(row[features])[0]
labels=["WORKABLE","CAUTION","HIGH_DELAY_RISK"]

a,b=st.columns(2)
a.metric("Predicted condition",labels[pred])
b.metric("Model confidence",f"{proba.max()*100:.1f}%")
st.subheader("Prediction probabilities")
st.bar_chart(pd.DataFrame({"Probability":proba},index=labels))
st.subheader("Entered conditions")
st.write({"rainfall_mm":rain,"soil_moisture":moisture,"operation":operation,"crop":crop,"stage":stage,"area_acres":area})
