from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from feature_engineering import build_features
from label_generation import create_proxy_labels

MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
MET_DIR = Path("results/metrics"); MET_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv("data/processed/environment.csv")
    df = create_proxy_labels(build_features(df)).sort_values("date").reset_index(drop=True)

    split = int(len(df)*0.80)
    train, test = df.iloc[:split], df.iloc[split:]
    target = "risk_label"
    drop = ["date","district","risk_label","risk_label_name","risk_score_proxy","recovery_class"]
    features = [c for c in train.columns if c not in drop]

    X_train,y_train = train[features],train[target]
    X_test,y_test = test[features],test[target]
    cat = X_train.select_dtypes(include=["object"]).columns.tolist()
    num = [c for c in features if c not in cat]

    prep = ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),
                         ("scale",StandardScaler())]),num),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                         ("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)
    ])

    models = {
        "logistic_regression": LogisticRegression(max_iter=1500,random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=400,max_depth=14,min_samples_leaf=3,
            class_weight="balanced",random_state=42,n_jobs=-1)
    }

    scores={}
    for name,estimator in models.items():
        pipe=Pipeline([("preprocessor",prep),("model",estimator)])
        pipe.fit(X_train,y_train)
        pred=pipe.predict(X_test)
        scores[name]={
            "accuracy":float(accuracy_score(y_test,pred)),
            "macro_f1":float(f1_score(y_test,pred,average="macro"))
        }
        joblib.dump(pipe,MODEL_DIR/f"{name}.joblib")
        print(name,scores[name])

    test.to_csv("data/processed/test_input.csv",index=False)
    with open(MET_DIR/"model_scores.json","w") as f: json.dump(scores,f,indent=2)

if __name__=="__main__":
    main()
