import joblib
import pandas as pd

LABELS={0:"WORKABLE",1:"CAUTION",2:"HIGH_DELAY_RISK"}

def predict_one(record):
    model=joblib.load("models/random_forest.joblib")
    x=pd.DataFrame([record])
    pred=int(model.predict(x)[0])
    proba=model.predict_proba(x)[0]
    return {
        "risk_label":LABELS[pred],
        "confidence":float(proba.max()),
        "probabilities":{LABELS[i]:float(p) for i,p in enumerate(proba)}
    }
