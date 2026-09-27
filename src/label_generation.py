import numpy as np

LABELS = {0:"WORKABLE",1:"CAUTION",2:"HIGH_DELAY_RISK"}

def create_proxy_labels(df):
    # Prototype-only: these are rule-derived proxy labels, not observed outcomes.
    x = df.copy()
    score = (
        0.40*np.clip(x["rainfall_3d"]/80,0,1)
        + 0.35*np.clip(x["soil_moisture"]/70,0,1)
        + 0.15*np.clip(x["rainfall_1d"]/40,0,1)
        + 0.10*np.clip(np.maximum(x["soil_moisture_change_1d"],0)/15,0,1)
    )
    score += np.where(x["operation_type"].eq("harvesting"),0.08,0)
    score += np.where(x["operation_type"].eq("spraying"),0.05,0)
    score = np.clip(score,0,1)

    x["risk_score_proxy"] = score
    x["risk_label"] = np.select([score<0.35,score<0.65],[0,1],default=2).astype(int)
    x["risk_label_name"] = x["risk_label"].map(LABELS)
    x["recovery_class"] = np.select(
        [score<0.35,score<0.65,score<0.80],[0,1,2],default=3
    ).astype(int)
    return x
