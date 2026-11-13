import pandas as pd, numpy as np
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data" / "patients"


def rd(name):
    p = D / f"{name}.csv"
    return pd.read_csv(p, low_memory=False) if p.exists() else None


def to_dt(s):
    return (
        pd.to_datetime(s, utc=True, errors="coerce")
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
    )


# ---------- Encounters (one row per encounter = one ML sample) ----------
enc = rd("encounters")
enc["START"] = to_dt(enc["START"])
enc = enc.dropna(subset=["START"]).sort_values("START").reset_index(drop=True)
enc["day"] = enc["START"].dt.normalize()

# ---------- Labels (y): risk_score in [0,1] -> risk_class ----------
base = {
    "wellness": 0.05,
    "ambulatory": 0.10,
    "outpatient": 0.15,
    "urgentcare": 0.35,
    "emergency": 0.60,
    "inpatient": 0.70,
}
enc["base"] = enc["ENCOUNTERCLASS"].map(base)
enc["cost"] = pd.to_numeric(
    enc.get("TOTAL_CLAIM_COST", enc.get("BASE_ENCOUNTER_COST")), errors="coerce"
)
enc = enc.dropna(subset=["base", "cost"]).reset_index(drop=True)

# relative cost within the same visit type (0..1) = how heavy this visit was
enc["cost_pct"] = enc.groupby("ENCOUNTERCLASS")["cost"].rank(pct=True)

# died within 30 days after the encounter
pat0 = rd("patients")
if "DEATHDATE" in pat0.columns:
    death = to_dt(pat0.set_index("Id")["DEATHDATE"])
    days = (enc["PATIENT"].map(death) - enc["START"]).dt.days
    enc["died30"] = days.between(0, 30).astype(int)
else:
    enc["died30"] = 0

enc["risk_score"] = (
    (enc["base"] + 0.15 * enc["cost_pct"] + 0.25 * enc["died30"]).clip(0, 1).round(4)
)
enc["risk_class"] = pd.cut(
    enc["risk_score"],
    bins=[-0.001, 0.25, 0.50, 0.75, 1.0],
    labels=["Low", "Medium", "High", "Critical"],
).astype(str)

# internal only (used by prior_urgent_visits below), not exported
enc["urgent"] = (
    enc["ENCOUNTERCLASS"].isin(["urgentcare", "emergency", "inpatient"]).astype(int)
)


# ---------- Helper: count of events strictly BEFORE the encounter day ----------
def prior_counts(patient_ids, dates, name):
    a = pd.DataFrame(
        {
            "PATIENT": enc["PATIENT"].values,
            "t": enc["day"].values,
            "o": 0,
            "row": enc.index.values,
        }
    )
    b = pd.DataFrame(
        {
            "PATIENT": np.asarray(patient_ids),
            "t": dates.dt.normalize().values,
            "o": 1,
            "row": -1,
        }
    )
    m = (
        pd.concat([a, b])
        .dropna(subset=["t"])
        .sort_values(["PATIENT", "t", "o"], kind="stable")
    )
    m["c"] = m.groupby("PATIENT")["o"].cumsum()
    enc[name] = m[m.o == 0].set_index("row")["c"].reindex(enc.index).fillna(0).values


# History of past urgent visits and total visits
prior_counts(
    enc.loc[enc.urgent == 1, "PATIENT"],
    enc.loc[enc.urgent == 1, "START"],
    "prior_urgent_visits",
)
prior_counts(enc["PATIENT"], enc["START"], "prior_visits")

# ---------- Event tables -> prior counts ----------
for name in [
    "conditions",
    "medications",
    "procedures",
    "immunizations",
    "allergies",
    "careplans",
    "imaging_studies",
    "devices",
]:
    t = rd(name)
    if t is None:
        continue
    col = "START" if "START" in t.columns else "DATE"
    prior_counts(t["PATIENT"], to_dt(t[col]), f"prior_{name}")

# Risk-factor flags from condition history
cond = rd("conditions")
cond["d"] = to_dt(cond["START"])
flags = {
    "hypertension": "hypertension",
    "diabetes": "diabetes|prediabetes",
    "heart_disease": "coronary|heart failure|myocardial|cardiac",
    "copd": "copd|emphysema|chronic obstructive",
    "obesity": "obes",
    "smoker": "tobacco|smok",
    "stroke": "stroke|cerebrovascular",
    "ckd": "chronic kidney",
    "cancer": "cancer|neoplasm|carcinoma",
    "asthma": "asthma",
}
for k, rx in flags.items():
    sub = cond[cond.DESCRIPTION.str.contains(rx, case=False, na=False)]
    prior_counts(sub["PATIENT"], sub["d"], f"rf_{k}")
    enc[f"rf_{k}"] = (enc[f"rf_{k}"] > 0).astype(int)

# ---------- Observations: latest numeric value BEFORE the encounter ----------
obs = rd("observations")
if "TYPE" in obs.columns:
    obs = obs[obs["TYPE"] == "numeric"]
obs["VALUE"] = pd.to_numeric(obs["VALUE"], errors="coerce")
obs = obs.dropna(subset=["VALUE"])
obs["DATE"] = to_dt(obs["DATE"])

n_pat = obs["PATIENT"].nunique()
cov = obs.groupby("DESCRIPTION")["PATIENT"].nunique() / n_pat
keep = cov[cov >= 0.20].index  # drop rare measurements
obs = obs[obs.DESCRIPTION.isin(keep)]

w = (
    obs.pivot_table(
        index=["PATIENT", "DATE"], columns="DESCRIPTION", values="VALUE", aggfunc="mean"
    )
    .reset_index()
    .sort_values("DATE")
)
vcols = [c for c in w.columns if c not in ("PATIENT", "DATE")]
w[vcols] = w.groupby("PATIENT")[vcols].ffill()  # carry last known value per measurement
w = w.rename(
    columns={c: "obs_" + c.strip().replace(" ", "_").replace("/", "_") for c in vcols}
)

df = pd.merge_asof(
    enc.sort_values("START"),
    w,
    left_on="START",
    right_on="DATE",
    by="PATIENT",
    direction="backward",
    allow_exact_matches=False,
)

# ---------- Patient demographics (raw, no encoding) ----------
pat = rd("patients")
pat["BIRTHDATE"] = to_dt(pat["BIRTHDATE"])
cat_cols = ["GENDER", "RACE", "ETHNICITY"]
pcols = ["Id", "BIRTHDATE"] + cat_cols + (["INCOME"] if "INCOME" in pat.columns else [])
df = df.merge(
    pat[pcols], left_on="PATIENT", right_on="Id", how="left", suffixes=("", "_pat")
)
df["age"] = (df["START"] - df["BIRTHDATE"]).dt.days / 365.25

# ---------- Final frame ----------
feature_cols = (
    ["age"]
    + (["INCOME"] if "INCOME" in df.columns else [])
    + cat_cols
    + [c for c in df.columns if c.startswith(("prior_", "rf_", "obs_"))]
)
out = df[["PATIENT"] + feature_cols + ["risk_score", "risk_class"]].copy()

obs_cols = [c for c in out.columns if c.startswith("obs_")]
out = out.dropna(subset=obs_cols, how="all")

out.to_csv(Path(__file__).resolve().parent / "ml" / "triage_dataset.csv", index=False)
print(out.shape)
print(out["risk_class"].value_counts(normalize=True).round(3))
print(out["risk_score"].describe())
