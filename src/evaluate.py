from pathlib import Path
import json, joblib, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from feature_engineering import build_features
from label_generation import create_proxy_labels

FIG=Path("results/figures"); MET=Path("results/metrics")
FIG.mkdir(parents=True,exist_ok=True); MET.mkdir(parents=True,exist_ok=True)

def main():
    df=create_proxy_labels(build_features(
        pd.read_csv("data/processed/environment.csv")
    )).sort_values("date").reset_index(drop=True)

    split=int(len(df)*0.80); test=df.iloc[split:]
    drop=["date","district","risk_label","risk_label_name","risk_score_proxy","recovery_class"]
    features=[c for c in test.columns if c not in drop]

    model=joblib.load("models/random_forest.joblib")
    pred=model.predict(test[features])

    report=classification_report(
        test["risk_label"],pred,
        target_names=["WORKABLE","CAUTION","HIGH_DELAY_RISK"],
        output_dict=True,zero_division=0
    )
    with open(MET/"random_forest_report.json","w") as f: json.dump(report,f,indent=2)

    cm=confusion_matrix(test["risk_label"],pred)
    disp=ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["WORKABLE","CAUTION","HIGH_DELAY_RISK"]
    )
    fig,ax=plt.subplots(figsize=(7,5))
    disp.plot(ax=ax,cmap="Blues",colorbar=False)
    ax.set_title("Random Forest confusion matrix")
    fig.tight_layout()
    fig.savefig(FIG/"random_forest_confusion_matrix.png",dpi=180)
    plt.close(fig)

    print(classification_report(
        test["risk_label"],pred,
        target_names=["WORKABLE","CAUTION","HIGH_DELAY_RISK"],
        zero_division=0
    ))

if __name__=="__main__":
    main()
