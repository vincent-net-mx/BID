import numpy as np, pandas as pd, catboost as cb
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from motor_bayesiano import MotorBayesiano
import predictor as P

df = pd.read_csv("BID_dataset.csv").drop(columns=P.COLUMNAS_PROHIBIDAS)
y = df[P.TARGET].astype(int).to_numpy()
FR = datetime(2026,7,28)

def cv(feats, splitter, grupos=None, leak_bayes=False):
    cats=[c for c in P.CAT_FEATURES if c in feats]
    accs,f1s,aucs=[],[],[]
    args=(df,y,grupos) if grupos is not None else (df,y)
    # motor "con fuga": ajustado sobre TODO el dataset (replica el bug original)
    motor_full = MotorBayesiano().fit(df, P.V_ACTIVAS, target=P.TARGET)
    p_full = motor_full.inferir_lote(df, P.V_ACTIVAS)
    for tr,te in splitter.split(*args):
        dtr,dte=df.iloc[tr].reset_index(drop=True),df.iloc[te].reset_index(drop=True)
        if leak_bayes:
            ptr,pte = p_full[tr], p_full[te]
        else:
            m=MotorBayesiano().fit(dtr,P.V_ACTIVAS,target=P.TARGET)
            ptr,pte=m.inferir_lote(dtr,P.V_ACTIVAS),m.inferir_lote(dte,P.V_ACTIVAS)
        Xtr=P.construir_features(dtr,ptr,FR)[feats]; Xte=P.construir_features(dte,pte,FR)[feats]
        mdl=cb.CatBoostClassifier(**P.PARAMS_CATBOOST); mdl.fit(Xtr,y[tr],cat_features=cats)
        pr=mdl.predict_proba(Xte)[:,1]
        accs.append(accuracy_score(y[te],(pr>.5).astype(int)))
        f1s.append(f1_score(y[te],(pr>.5).astype(int),zero_division=0))
        aucs.append(roc_auc_score(y[te],pr))
    return np.mean(accs),np.std(accs,ddof=1),np.mean(f1s),np.std(f1s,ddof=1),np.mean(aucs)

skf=StratifiedKFold(5,shuffle=True,random_state=42)
sgkf=StratifiedGroupKFold(5,shuffle=True,random_state=42)
gr=df["asn"].astype(str).to_numpy()

ALL=P.FEATURES
SIN_BAYES=[f for f in ALL if f!="bayesian_risk"]
SIN_ID=[f for f in ALL if f not in ("asn","isp","infra_owner")]
SOLO_ABUSE=["abuseip_score","attack_type_count","abuseip_distinct_users","days_since_last_report","usage_type"]

print(f"{'Configuración':<42}{'Acc±std':<20}{'F1±std':<20}{'AUC'}")
print("-"*100)
for nom,feats,sp,g,leak in [
  ("A. Completo + Bayes con FUGA (original)",ALL,skf,None,True),
  ("B. Completo + Bayes fold-safe",ALL,skf,None,False),
  ("C. Sin bayesian_risk",SIN_BAYES,skf,None,False),
  ("D. Sin asn/isp/infra_owner",SIN_ID,skf,None,False),
  ("E. Solo señales AbuseIPDB",SOLO_ABUSE,skf,None,False),
  ("F. Completo, split por ASN (grupo)",ALL,sgkf,gr,False),
  ("G. Sin identificadores, split por ASN",SIN_ID,sgkf,gr,False),
]:
    a,asd,f,fsd,auc=cv(feats,sp,g,leak)
    print(f"{nom:<42}{a:.4f}±{asd:.4f}     {f:.4f}±{fsd:.4f}     {auc:.4f}")
