"""Demo of the triage policy.  Run: python -m streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st

from triage import COST, HOSPITALS, N, STAGES, load, next_test, train

st.set_page_config(page_title="Heart test triage", page_icon="🫀")


@st.cache_resource(show_spinner="Training on all 4 hospitals...")
def trained():
    data = [load(h) for h in HOSPITALS]
    return train(pd.concat([d[0] for d in data]), np.concatenate([d[1] for d in data]))


def pick(label, options, key):
    return st.selectbox(label, list(options), format_func=options.get, key=key)


YES_NO = {0: "No", 1: "Yes"}
INPUTS = {
    "age":      lambda: st.number_input("Age", 20, 100, 55),
    "sex":      lambda: pick("Sex", {1: "Male", 0: "Female"}, "sex"),
    "cp":       lambda: pick("Chest pain type", {1: "Typical angina", 2: "Atypical angina",
                                                 3: "Non-anginal pain", 4: "No chest pain"}, "cp"),
    "trestbps": lambda: st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130),
    "chol":     lambda: st.number_input("Serum cholesterol (mg/dl)", 100, 600, 240),
    "fbs":      lambda: pick("Fasting blood sugar > 120 mg/dl", YES_NO, "fbs"),
    "restecg":  lambda: pick("Resting ECG", {0: "Normal", 1: "ST-T wave abnormality",
                                             2: "Left ventricular hypertrophy"}, "restecg"),
    "exang":    lambda: pick("Angina during exercise", YES_NO, "exang"),
    "oldpeak":  lambda: st.number_input("ST depression, exercise vs rest (mm)", 0.0, 7.0, 1.0, 0.1),
    "slope":    lambda: pick("Peak exercise ST slope", {1: "Upsloping", 2: "Flat", 3: "Downsloping"}, "slope"),
    "ca":       lambda: pick("Major vessels coloured by fluoroscopy", {0: "0", 1: "1", 2: "2", 3: "3"}, "ca"),
    "thalach":  lambda: st.number_input("Max heart rate reached", 60, 220, 150),
    "thal":     lambda: pick("Thallium scan", {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}, "thal"),
}

t = trained()
st.title("Which heart test next?")
st.caption("Research prototype on the 1988 UCI heart disease data (920 patients). Not medical advice.")

row, m, spent, s = {}, 0, 0.0, 0
while True:
    name, fields, cost = STAGES[s]
    st.subheader(f"{name}  ·  ${cost:.2f}")
    for f in fields:
        row[f] = INPUTS[f]()
    m, spent = m | 1 << s, spent + cost
    p = t["model"](list(row)).predict_proba(pd.DataFrame([row]))[0, 1]
    confidence = max(p, 1 - p)
    st.progress(float(p), text=f"Chance of coronary artery disease: {p:.0%}")

    if confidence >= t["conf"]:
        verdict = "likely disease" if p >= 0.5 else "unlikely to have disease"
        st.success(f"Confident enough to stop: **{verdict}** ({confidence:.0%} sure). "
                   f"Tests used: ${spent:.2f} of $323.97 for the full workup.")
        break
    s = next_test(t["ref"], m, p, np.ones(N, bool))
    if s is None:
        st.warning(f"Still only {confidence:.0%} sure after every test: refer to a cardiologist.")
        break
    st.info(f"Only {confidence:.0%} sure (needs {t['conf']:.0%}). "
            f"Recommended next: **{STAGES[s][0]}** (${COST[s]:.2f}).")
    if not st.checkbox(f"I have the {STAGES[s][0]} results", key=f"got{s}"):
        break
