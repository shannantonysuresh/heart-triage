"""Cost-aware heart disease triage.

For each patient: start with the $4 intake, then keep ordering the test with the best
expected error reduction per dollar until the model is confident. If every available test
is done and it still isn't confident, the patient is referred (no automatic diagnosis).

Run: python triage.py   -> prints results, writes results.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
HOSPITALS = ["cleveland", "hungarian", "va", "switzerland"]
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
CATEGORICAL = {"cp", "restecg", "slope", "thal"}

# CAD prices from data/costs/heart-disease.expense: the first test in a group pays full
# price, the rest of that group are discounted. Groups never span two stages.
STAGES = [
    ("intake",            ["age", "sex", "cp", "trestbps"], 4 * 1.00),
    ("bloods + rest ECG", ["chol", "fbs", "restecg"],       7.27 + 3.10 + 15.50),
    ("exercise ECG",      ["exang", "oldpeak", "slope"],    87.30 + 1.00 + 1.00),
    ("fluoroscopy",       ["ca"],                           100.90),
    ("thallium scan",     ["thalach", "thal"],              102.90 + 1.00),
]
N = len(STAGES)
COST = np.array([s[2] for s in STAGES])
MASKS = [m for m in range(1 << N) if m & 1]  # sets of stages done; intake is always first
MASK_ALL = (1 << N) - 1
CONFS = np.round(np.arange(0.50, 0.99, 0.02), 2)
TARGET = 0.95   # accuracy required on the patients the system decides
MIN_ROWS = 50   # fewer training patients than this with a test -> the test can't be modelled
K = 25          # reference patients used to estimate a test's value


def feats(mask, usable):
    return [f for k, s in enumerate(STAGES) if mask >> k & 1 for f in s[1] if f in usable]


def load(hospital):
    df = pd.read_csv(HERE / "data" / f"processed.{hospital}.data", names=COLS, na_values="?")
    df[["chol", "trestbps"]] = df[["chol", "trestbps"]].replace(0, np.nan)  # 0 = missing (Swiss/VA)
    return df.drop(columns="num"), (df.num > 0).astype(int).to_numpy()


def make_model(cols):
    num = [f for f in cols if f not in CATEGORICAL]
    cat = [f for f in cols if f in CATEGORICAL]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), cat),
    ])
    # Logistic regression: beat random forest and gradient boosting in CV (see README).
    return make_pipeline(pre, LogisticRegression(C=0.3, max_iter=2000))


def availability(X, usable):
    """avail[i, stage]: patient i had at least one modellable result from that stage."""
    return np.column_stack([X[[f for f in s[1] if f in usable]].notna().any(axis=1).to_numpy()
                            for s in STAGES])


def make_ref(P):
    """Reference patients' stage probabilities, pre-sorted so the K closest are a slice away."""
    order = np.argsort(P, axis=0)
    return {"P": P, "err": np.minimum(P, 1 - P), "order": order, "sorted": np.take_along_axis(P, order, 0)}


def train(X, y):
    """Everything the policy needs from one training set."""
    X = X.reset_index(drop=True)
    usable = [c for c in X.columns if X[c].notna().sum() >= MIN_ROWS]
    models = {}

    def model(cols):
        """Logistic model trained only on patients who had exactly these tests recorded."""
        if tuple(cols) not in models:
            ok = X[cols].notna().all(axis=1).to_numpy()
            if ok.sum() < MIN_ROWS:  # rare combination: impute rather than train on a handful
                ok = np.ones(len(X), bool)
            models[tuple(cols)] = make_model(cols).fit(X.loc[ok, cols], y[ok])
        return models[tuple(cols)]

    # Out-of-fold probabilities on the training set, averaged over 3 CV shuffles so the
    # threshold and test choice don't hinge on one lucky split.
    P = np.full((len(X), 1 << N), np.nan)
    for m in MASKS:
        f = feats(m, usable)
        P[:, m] = np.mean([cross_val_predict(make_model(f), X[f], y, method="predict_proba",
                                             cv=StratifiedKFold(5, shuffle=True, random_state=r))[:, 1]
                           for r in range(3)], axis=0)
    ref = make_ref(P)
    return {"model": model, "usable": usable, "ref": ref,
            "conf": pick_conf(ref, availability(X, usable), y)}


def predict(t, X):
    """P[i, mask] = P(disease) after the stages in `mask`, using only values recorded for patient i."""
    X = X.reset_index(drop=True)
    P = np.full((len(X), 1 << N), np.nan)
    for m in MASKS:
        f = feats(m, t["usable"])
        for pattern, grp in X[f].notna().groupby(f):
            cols = [c for c, known in zip(f, pattern) if known]
            P[grp.index, m] = t["model"](cols).predict_proba(X.loc[grp.index, cols])[:, 1]
    return P, availability(X, t["usable"])


def gain(ref, m, s, p):
    """Expected drop in error from adding stage s, measured on the K reference patients whose
    current probability is closest to p. ponytail: neighbours on p only, not on raw features."""
    lo = np.clip(np.searchsorted(ref["sorted"][:, m], p) - K // 2, 0, len(ref["P"]) - K)
    nb = ref["order"][lo:lo + K, m]
    return ref["err"][nb, m].mean() - ref["err"][nb, m | 1 << s].mean()


def next_test(ref, m, p, avail_row):
    """Best remaining stage by expected error drop per dollar, or None if nothing is left."""
    cands = [s for s in range(1, N) if not m >> s & 1 and avail_row[s]]
    return max(cands, key=lambda s: gain(ref, m, s, p) / COST[s]) if cands else None


def run_policy(P, avail, conf, ref, greedy=True):
    """Returns final p, cost and whether the system decided (else: referred) per patient."""
    final, cost, decided = np.empty(len(P)), np.empty(len(P)), np.zeros(len(P), bool)
    for i in range(len(P)):
        m, c = 1, COST[0]
        while True:
            p = P[i, m]
            if max(p, 1 - p) >= conf:
                decided[i] = True
                break
            s = next_test(ref, m, p, avail[i]) if greedy else \
                next((s for s in range(1, N) if not m >> s & 1 and avail[i, s]), None)
            if s is None:
                break
            m, c = m | 1 << s, c + COST[s]
        final[i], cost[i] = p, c
    return final, cost, decided


def score(final, cost, decided, y):
    right = (final >= 0.5) == y
    return {"acc": right.mean(), "cost": cost.mean(), "coverage": decided.mean(),
            "decided_acc": right[decided].mean() if decided.any() else np.nan}


def pick_conf(ref, avail, y):
    """Lowest confidence (= most patients decided) whose decided accuracy reaches TARGET."""
    for c in CONFS:
        if score(*run_policy(ref["P"], avail, c, ref), y)["decided_acc"] >= TARGET:
            return c
    return CONFS[-1]


def all_tests(P, avail):
    """Each patient's probability after every test they actually had."""
    return P[np.arange(len(P)), avail @ (1 << np.arange(N))]


def wilson(k, n, z=1.96):
    c, h = (k + z * z / 2) / (n + z * z), z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return c - h, c + h


def cleveland_cv(data, pooled, n_repeats=3):
    """Test on Cleveland folds; train on the other Cleveland folds (+ the other 3 hospitals if pooled)."""
    X, y = data["cleveland"]
    extra = [data[h] for h in HOSPITALS[1:]] if pooled else []
    policies = ("best next test", "fixed order") if pooled else ("best next test",)
    curves = {p: [] for p in policies}
    acc_all, tally, confs = [], np.zeros(4), []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=0).split(X, y):
        t = train(pd.concat([X.iloc[tr]] + [e[0] for e in extra]),
                  np.concatenate([y[tr]] + [e[1] for e in extra]))
        P, avail = predict(t, X.iloc[te])
        for name in policies:
            curves[name].append([score(*run_policy(P, avail, c, t["ref"], name == "best next test"), y[te])
                                 for c in CONFS])
        acc_all.append(((all_tests(P, avail) >= 0.5) == y[te]).mean())
        final, cost, d = run_policy(P, avail, t["conf"], t["ref"])
        tally += [d.sum(), ((final >= 0.5) == y[te])[d].sum(), len(te), cost.sum()]
        confs.append(t["conf"])
    curves = {n: {k: np.nanmean([[r[k] for r in split] for split in v], axis=0)
                  for k in ("acc", "cost", "coverage", "decided_acc")} for n, v in curves.items()}
    return curves, np.array(acc_all), tally, np.array(confs)


def main():
    data = {h: load(h) for h in HOSPITALS}
    X, y = data["cleveland"]
    print(f"Cleveland: {len(y)} patients, {y.mean():.0%} with disease. CV = 5-fold x3, tested on Cleveland.\n")

    runs = {"Cleveland only": cleveland_cv(data, pooled=False), "all 4 hospitals": cleveland_cv(data, pooled=True)}
    for name, (curves, acc_all, (n_dec, n_right, n, cost), confs) in runs.items():
        lo, hi = wilson(n_right, n_dec)
        bn = curves["best next test"]
        i = np.argmax(bn["acc"] >= acc_all.mean() - 0.01)
        print(f"Trained on {name}:")
        print(f"  all tests, everyone decided   acc {acc_all.mean():.3f} ± {acc_all.std():.3f}   $323.97")
        print(f"  best next test, within 1pt    acc {bn['acc'][i]:.3f}   ${bn['cost'][i]:.2f}")
        if "fixed order" in curves:
            fo = curves["fixed order"]
            j = np.argmax(fo["acc"] >= acc_all.mean() - 0.01)
            print(f"  fixed order, within 1pt       acc {fo['acc'][j]:.3f}   ${fo['cost'][j]:.2f}")
        print(f"  decide-or-refer: {n_right / n_dec:.3f} correct (95% CI {lo:.3f}-{hi:.3f}) on "
              f"{n_dec / n:.0%} of patients, ${cost / n:.0f} avg, conf {confs.mean():.2f} ± {confs.std():.2f}\n")

    leave_one_hospital_out(data)
    plot(runs)


def leave_one_hospital_out(data):
    print("Leave-one-hospital-out (vs. trained on Cleveland only):")
    print(f"  {'test on':<12}{'train on':<16}{'rate':>6}{'majority':>10}{'acc':>7}{'AUC':>7}"
          f"{'decided':>9}{'dec acc':>9}{'cost':>7}  tests it can't use")
    for h in HOSPITALS:
        options = [("other 3", [o for o in HOSPITALS if o != h])] + ([("Cleveland", ["cleveland"])] if h != "cleveland" else [])
        for label, src in options:
            t = train(pd.concat([data[o][0] for o in src]), np.concatenate([data[o][1] for o in src]))
            Xh, yh = data[h]
            P, avail = predict(t, Xh)
            full = all_tests(P, avail)
            s = score(*run_policy(P, avail, t["conf"], t["ref"]), yh)
            missing = [c for c in COLS[:-1] if c not in t["usable"]]
            print(f"  {h:<12}{label:<16}{yh.mean():>6.0%}{max(yh.mean(), 1 - yh.mean()):>10.3f}"
                  f"{((full >= .5) == yh).mean():>7.3f}{roc_auc_score(yh, full):>7.3f}{s['coverage']:>9.0%}"
                  f"{s['decided_acc']:>9.3f}{'$%.0f' % s['cost']:>7}  {', '.join(missing) or '-'}")


def plot(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    styles = {("Cleveland only", "best next test"): ("#8a8f98", "-"),
              ("all 4 hospitals", "best next test"): ("#2a6fdb", "-"),
              ("all 4 hospitals", "fixed order"): ("#2a6fdb", ":")}
    for (run, pol), (color, ls) in styles.items():
        cv = runs[run][0][pol]
        a1.plot(cv["cost"], cv["acc"], color=color, ls=ls, lw=2, label=f"trained on {run}, {pol}")
        if pol == "best next test":
            a2.plot(cv["coverage"], cv["decided_acc"], color=color, lw=2, marker="o", ms=3, label=f"trained on {run}")
    a1.set(xlabel="Average test cost per patient (CAD)", ylabel="Accuracy, everyone decided",
           title="Cost vs accuracy (tested on Cleveland)")
    a2.axhline(TARGET, color="#d9480f", lw=1, ls="--", label=f"{TARGET:.0%} target")
    a2.set(xlabel="Share of patients decided (rest referred)", ylabel="Accuracy on decided patients",
           title="Decide-or-refer trade-off")
    for a in (a1, a2):
        a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False); a.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "results.png", dpi=150)
    print("\nwrote results.png")


def _selfcheck():
    assert abs(COST.sum() - 323.97) < 0.01  # full workup cost with group discounts
    # patient 0 confident at intake; patient 1 has no other tests available -> referred
    P = np.full((K, 1 << N), 0.5); P[0, 1] = 0.95
    avail = np.zeros((K, N), bool); avail[:, 0] = True; avail[0] = True
    final, cost, decided = run_policy(P[:2], avail[:2], 0.8, make_ref(P))
    assert decided.tolist() == [True, False] and cost.tolist() == [4.0, 4.0]
    lo, hi = wilson(95, 100)
    assert lo < 0.95 < hi and hi < 1


if __name__ == "__main__":
    _selfcheck()
    main()
