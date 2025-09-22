
import os
import numpy as np
import pandas as pd
import streamlit as st

# Optional: set PyTensor flags externally if needed (e.g., avoid C++ JIT on Windows)
# os.environ["PYTENSOR_FLAGS"] = "cxx=,mode=FAST_COMPILE,optimizer=None"

# Optional PyMC imports
try:
    import pymc as pm
    import arviz as az
    _PYMC_AVAILABLE = True
except Exception:
    pm = None
    az = None
    _PYMC_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
)

# CalibrationDisplay may not exist in older scikit-learn versions
try:
    from sklearn.calibration import CalibrationDisplay
    _HAS_CALIB = True
except Exception:
    CalibrationDisplay = None
    _HAS_CALIB = False

import matplotlib.pyplot as plt

# ========= Page config =========
st.set_page_config(page_title="Student Performance — Logistic (MLE) & Bayesian", layout="wide")

st.title("🎓 Student Performance — Logistic (MLE) & Bayesian (Laplace / PyMC)")
st.caption("Upload CSV → auto-detect target → Logistic (MLE) + Bayesian (Laplace or PyMC). Compact plots, rich explanations, parameter guides, threshold tuning, and lift/gains.")

# ========= Quick Start =========
with st.expander("🚀 Quick Start (Read me first)", expanded=False):
    st.markdown("""
**Steps**
1. **Upload** your CSV (rows = students, columns = features + target).  
2. The app **auto-detects** the **target**; adjust if needed. Continuous targets are **auto-binarized** with a balanced threshold.
3. Adjust sidebar **parameters** (with tooltips).
4. Review **Descriptive** → class balance & correlations.
5. Train **Logistic (MLE)** and **Bayesian** (Laplace or PyMC).  
6. Use **Threshold & Gains** to tune the decision threshold, and **Predict** to try custom inputs.

**Tip:** For imbalanced data, start with `class_weight='balanced'` and tune the threshold for your business goal (e.g., recall).
""")

# ========= Sidebar: global controls =========
st.sidebar.header("⚙️ Global Settings")
test_size = st.sidebar.slider("Test size", 0.10, 0.40, 0.25, 0.01,
                              help="Phần trăm dữ liệu dành cho kiểm tra (test). 0.25 = 25% test, 75% train.")
random_state = st.sidebar.number_input("Random seed", value=42, step=1,
                                       help="Giữ cố định để tái lập kết quả chia train/test.")
plot_scale = st.sidebar.slider("Plot size scale", 0.6, 1.6, 0.9, 0.05,
                               help="Thu/phóng toàn bộ biểu đồ để gọn gàng hơn.")

# Apply scaled font sizes
plt.rcParams.update({
    "axes.titlesize": 12*plot_scale,
    "axes.labelsize": 11*plot_scale,
    "xtick.labelsize": 10*plot_scale,
    "ytick.labelsize": 10*plot_scale,
    "legend.fontsize": 10*plot_scale,
})

def make_fig(w=5, h=3):
    """Create a small figure scaled by plot_scale."""
    fig, ax = plt.subplots(figsize=(w*plot_scale, h*plot_scale))
    return fig, ax

# ========= Parameter Guide =========
with st.expander("📘 Parameter Guide & How They Affect Predictions", expanded=False):
    st.markdown("""
### Data split
- **Test size**: phần trăm dữ liệu giữ lại để đo lường (không dùng để huấn luyện).

### Logistic (MLE)
- **Penalty**
  - **None (MLE)** = không phạt → ước lượng cực đại khả năng (tối đa hóa likelihood).
  - **L2** (Ridge) → **co** hệ số về 0 → ổn định, chống overfit.
- **C (khi L2)**: **C lớn → phạt yếu** (gần MLE). **C nhỏ → phạt mạnh** (xác suất “dịu” hơn, ít cực trị).
- **Solver**
  - **lbfgs**: ổn định, nhanh cho dữ liệu vừa.
  - **saga**: linh hoạt, tốt cho dữ liệu thưa/lớn.
- **Class weight**
  - **balanced** đối phó lệch lớp: thường tăng **Recall** lớp thiếu nhưng có thể giảm **Precision**.

### Bayesian (Laplace)
- **Prior σ (weights/intercept)**: độ lệch chuẩn prior Gaussian.
  - **σ lớn** → prior yếu (hệ số linh hoạt).
  - **σ nhỏ** → prior mạnh (co về 0). Kết quả **bảo thủ** hơn.

### Bayesian (PyMC sampling)
- **draws/tune**: nhiều hơn → posterior mượt hơn, chậm hơn.
- **target_accept**: 0.9–0.95 an toàn hơn, chậm hơn.
- **chains/cores**: nhiều chains để kiểm tra hội tụ.

### Ảnh hưởng lên dự đoán
- Regularization mạnh (L2 với **C nhỏ**, hoặc **σ nhỏ**) → hệ số nhỏ hơn → xác suất dự đoán **ít cực trị** (gần 0.5), mô hình **bảo thủ**.
- **balanced** giúp bắt được nhiều lớp thiểu số (**Recall↑**) nhưng có thể hi sinh **Precision**.
- Priors (Bayesian) cung cấp **độ bất định** (uncertainty) để ra quyết định dựa trên rủi ro.
""")

# ========= Helpers =========
CANDIDATE_TARGET_NAMES = ["Score","score","grade","passed","pass","target","label","y","outcome","result","performance"]

def safe_read_csv(file) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "cp1258"]
    seps_to_try = [",", ";", "\t", "|"]
    for enc in encodings_to_try:
        for sep in seps_to_try:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc, sep=sep)
                return df
            except Exception:
                continue
    st.error("❌ Không đọc được CSV với các encoding/sep phổ biến.")
    st.stop()

def infer_target_column(df: pd.DataFrame):
    cols_lower = {c.lower(): c for c in df.columns}
    for name in CANDIDATE_TARGET_NAMES:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return df.columns[-1]

def is_binary_series(s: pd.Series) -> bool:
    vals = pd.unique(s.dropna())
    try:
        vals = set([int(v) if str(v).isdigit() else v for v in vals])
    except Exception:
        vals = set(vals)
    return vals.issubset({0,1,True,False})

def binarize_continuous(y: pd.Series):
    vals = pd.to_numeric(y, errors="coerce")
    lo, hi = vals.quantile(0.2), vals.quantile(0.8)
    candidates = np.linspace(lo, hi, 51)
    best_thr, best_balance = candidates[0], -1.0
    for thr in candidates:
        y_tmp = (vals >= thr).astype(int)
        balance = 1 - abs(y_tmp.mean() - 0.5)*2
        if balance > best_balance:
            best_balance, best_thr = balance, thr
    return (vals >= best_thr).astype(int), float(best_thr), float((vals >= best_thr).mean())

def summarize_metrics(y_true, y_pred, y_proba=None):
    out = {}
    out["Accuracy"]  = accuracy_score(y_true, y_pred)
    out["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["Recall"]    = recall_score(y_true, y_pred, zero_division=0)
    out["F1"]        = f1_score(y_true, y_pred, zero_division=0)
    if y_proba is not None:
        try:
            out["ROC AUC"] = roc_auc_score(y_true, y_proba)
        except Exception:
            out["ROC AUC"] = None
    return out

def to_dense(X):
    try:
        return X.toarray()
    except Exception:
        return np.asarray(X)

def build_design_matrix(preprocess, X_df):
    Z = preprocess.transform(X_df)
    return to_dense(Z)

def laplace_posterior(Z, w, b, prior_var_w, prior_var_b):
    n, d = Z.shape
    eta = Z @ w + b
    p = 1.0 / (1.0 + np.exp(-eta))
    W = p * (1 - p)
    Z_aug = np.hstack([np.ones((n,1)), Z])
    prior_prec = np.diag(np.concatenate([[1.0/prior_var_b], np.full(d, 1.0/prior_var_w)]))
    A = (Z_aug.T * W) @ Z_aug
    precision = A + prior_prec
    try:
        cov = np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(precision)
    return cov

def bayes_pred_prob_with_variance_correction(m, v):
    # Approx E[sigmoid(N(m,v))] ≈ sigmoid( m / sqrt(1 + pi*v/8) )
    denom = np.sqrt(1.0 + np.pi * v / 8.0)
    return 1.0 / (1.0 + np.exp(-m/denom))

def make_mle_logreg(class_weight, solver, C, penalty):
    try:
        return LogisticRegression(
            penalty=penalty, C=C, solver=solver, max_iter=1000, class_weight=class_weight
        )
    except Exception:
        # Fallback emulate MLE via large C L2
        return LogisticRegression(
            penalty="l2", C=1e9, solver="lbfgs", max_iter=1000, class_weight=class_weight
        )

# ========= Sidebar: model-specific controls =========
st.sidebar.subheader("Logistic (MLE)")
penalty_opt = st.sidebar.selectbox("Penalty", ["None (MLE)", "L2"], index=0,
                                   help="None = không phạt (MLE). L2 = Ridge (co hệ số về 0).")
solver_opt = st.sidebar.selectbox("Solver", ["lbfgs", "saga"], index=0,
                                  help="lbfgs: ổn định, nhanh. saga: linh hoạt, tốt cho dữ liệu thưa.")
C_val = st.sidebar.number_input("C (nếu L2, lớn ~ MLE)", min_value=1e-4, max_value=1e12, value=1e6, step=1e4, format="%.4f",
                                help="C lớn → phạt yếu; C nhỏ → phạt mạnh.")
cw_opt = st.sidebar.selectbox("Class weight", ["None", "balanced"], index=1,
                              help="balanced giúp với lệch lớp, thường tăng Recall nhưng giảm Precision.")

st.sidebar.subheader("Bayesian method")
bayes_method = st.sidebar.selectbox("Chọn phương pháp", ["Laplace (nhanh)", "PyMC sampling"], index=0,
                                    help="Laplace: nhanh, không cần compiler. PyMC: posterior chính xác hơn nhưng cần toolchain.")
st.sidebar.caption("💡 Nếu PyMC lỗi biên dịch trên Windows, app sẽ tự fallback sang Laplace.")

with st.sidebar.expander("Bayesian (Laplace) params", expanded=(bayes_method=="Laplace (nhanh)")):
    prior_sigma_w = st.slider("Prior σ (weights)", 0.1, 100.0, 10.0, 0.1,
                              help="σ lớn → prior yếu (ít co). σ nhỏ → prior mạnh (co nhiều).")
    prior_sigma_b = st.slider("Prior σ (intercept)", 0.1, 100.0, 10.0, 0.1,
                              help="Tương tự cho intercept.")

with st.sidebar.expander("Bayesian (PyMC) params", expanded=(bayes_method=="PyMC sampling")):
    draws = st.number_input("draws", min_value=200, max_value=3000, value=1000, step=50,
                            help="Số mẫu posterior sau warmup.")
    tune = st.number_input("tune", min_value=200, max_value=3000, value=1000, step=50,
                           help="Số bước warmup (điều chỉnh NUTS).")
    target_accept = st.slider("target_accept", 0.80, 0.98, 0.90, 0.01,
                              help="Cao hơn → ít bước bị từ chối hơn nhưng chậm hơn.")
    chains = st.number_input("chains", min_value=1, max_value=4, value=2, step=1,
                             help="Số chuỗi độc lập để kiểm tra hội tụ.")
    cores = st.number_input("cores", min_value=1, max_value=2, value=1, step=1,
                            help="Số lõi CPU để chạy song song.")
    prior_sigma_beta_pymc = st.number_input("prior_sigma_beta", min_value=0.1, max_value=100.0, value=10.0, step=0.1,
                                            help="Prior σ cho hệ số (PyMC).")
    prior_sigma_intercept_pymc = st.number_input("prior_sigma_intercept", min_value=0.1, max_value=100.0, value=10.0, step=0.1,
                                                 help="Prior σ cho intercept (PyMC).")

# ========= Upload & target =========
file = st.file_uploader("📂 Upload CSV", type=["csv"])
if not file:
    st.info("Hãy upload file CSV để bắt đầu.")
    st.stop()

df = safe_read_csv(file)
st.subheader("🔎 Preview dữ liệu")
st.dataframe(df.head())

default_target = "Score" if "Score" in df.columns else infer_target_column(df)
target_col = st.selectbox("🎯 Chọn cột target", options=list(df.columns), index=list(df.columns).index(default_target))

y_raw = df[target_col]
X_raw = df.drop(columns=[target_col])

if is_binary_series(y_raw):
    st.success(f"Target `{target_col}` đã nhị phân → dùng trực tiếp (0/1).")
    y = y_raw.astype(int).replace({True:1, False:0})
    selected_threshold = None
else:
    y, selected_threshold, cls1_rate = binarize_continuous(y_raw)
    st.warning(f"`{target_col}` là liên tục → **auto-binarize** tại ngưỡng **{selected_threshold:.4f}** (tỷ lệ lớp 1 ≈ {cls1_rate:.2%}).")
    st.caption("Ngưỡng được tìm để cân bằng gần 50/50 giúp mô hình ổn định hơn. Bạn có thể tinh chỉnh ngưỡng trong mục **Threshold & Gains**.")

# ========= Split & preprocess =========
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=test_size, random_state=random_state, stratify=y
)

numeric_cols = [c for c in X_raw.columns if np.issubdtype(df[c].dtype, np.number)]
categorical_cols = [c for c in X_raw.columns if c not in numeric_cols]

numeric_tf = Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())])
categorical_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocess = ColumnTransformer([
    ("num", numeric_tf, numeric_cols),
    ("cat", categorical_tf, categorical_cols)
])
preprocess.fit(X_train)

# ========= Data Health Checks =========
with st.expander("🩺 Data Health Checks (types • missing • cardinality)", expanded=False):
    info = []
    for c in X_raw.columns:
        s = X_raw[c]
        missing = float(s.isna().mean()) if len(s) else 0.0
        unique = int(s.nunique(dropna=True))
        dtype = str(s.dtype)
        row = {"column": c, "dtype": dtype, "%missing": round(missing*100,2), "unique": unique}
        if np.issubdtype(s.dtype, np.number):
            try:
                row.update({"mean": float(pd.to_numeric(s, errors="coerce").mean()),
                            "std": float(pd.to_numeric(s, errors="coerce").std())})
            except Exception:
                pass
        else:
            try:
                top_vals = pd.Series(s.astype(str)).value_counts().head(5).index.tolist()
                row["top_values"] = ", ".join(map(str, top_vals))
            except Exception:
                row["top_values"] = ""
        info.append(row)
    st.dataframe(pd.DataFrame(info))

    st.caption("**Gợi ý**: %missing cao → cần thu thập thêm dữ liệu hoặc cân nhắc kỹ chiến lược impute. Cardinality quá lớn ở categorical → one-hot có thể nở chiều.")

# ========= Descriptive (smaller + explanations) =========
st.header("🧭 Descriptive")

fig_bal, ax_bal = make_fig(4.5, 3.0)
y_counts = pd.Series(y).value_counts().sort_index()
ax_bal.bar(y_counts.index.astype(str), y_counts.values)
ax_bal.set_title("Class Balance")
ax_bal.set_xlabel("Class")
ax_bal.set_ylabel("Count")
st.pyplot(fig_bal, use_container_width=True)
st.caption("**Ý nghĩa**: Kiểm tra độ cân bằng giữa hai lớp. Lệch lớp mạnh → cân nhắc `class_weight='balanced'` và điều chỉnh ngưỡng.")

if len(numeric_cols) >= 2:
    try:
        corr = pd.DataFrame(X_train[numeric_cols]).corr()
        fig_corr, ax_corr = make_fig(5.0, 3.2)
        im = ax_corr.imshow(corr, aspect='auto')
        ax_corr.set_xticks(range(len(numeric_cols))); ax_corr.set_xticklabels(numeric_cols, rotation=90)
        ax_corr.set_yticks(range(len(numeric_cols))); ax_corr.set_yticklabels(numeric_cols)
        ax_corr.set_title("Correlation (numeric)")
        fig_corr.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
        st.pyplot(fig_corr, use_container_width=True)
        st.caption("**Giải thích**: Tương quan cao giữa biến số có thể làm hệ số khó diễn giải; cân nhắc regularization hoặc PCA/feature selection.")
    except Exception:
        st.info("Không vẽ được heatmap tương quan.")

# ========= Logistic (MLE) =========
st.header("⚙️ Logistic Regression (MLE)")
class_weight = None if cw_opt == "None" else "balanced"
penalty = None if penalty_opt.startswith("None") else "l2"
solver = solver_opt
C_for_model = float(C_val)

mle_clf = make_mle_logreg(class_weight, solver, C_for_model, penalty)
mle_pipe = Pipeline([("preprocess", preprocess), ("clf", mle_clf)])
mle_pipe.fit(X_train, y_train)

y_pred_mle = mle_pipe.predict(X_test)
try:
    y_proba_mle = mle_pipe.predict_proba(X_test)[:, 1]
except Exception:
    y_proba_mle = None

mle_metrics = summarize_metrics(y_test, y_pred_mle, y_proba_mle)

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Accuracy", f"{mle_metrics['Accuracy']:.3f}")
with c2: st.metric("Precision", f"{mle_metrics['Precision']:.3f}")
with c3: st.metric("Recall", f"{mle_metrics['Recall']:.3f}")
with c4: st.metric("F1", f"{mle_metrics['F1']:.3f}")
if y_proba_mle is not None and mle_metrics.get("ROC AUC") is not None:
    st.metric("ROC AUC", f"{mle_metrics['ROC AUC']:.3f}")
st.caption("**Gợi ý**: Recall thấp cho lớp 1? → dùng `balanced` và/hoặc giảm ngưỡng. Precision thấp? → tăng ngưỡng.")

cm = confusion_matrix(y_test, y_pred_mle)
st.write("**Confusion Matrix (MLE)**")
st.write(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))
st.caption("**TN/FP/FN/TP**: False Positive tốn chi phí do cảnh báo sai; False Negative tốn chi phí do bỏ sót.")

if y_proba_mle is not None:
    colA, colB, colC = st.columns(3)
    with colA:
        fig, ax = make_fig(4.0, 3.0)
        RocCurveDisplay.from_predictions(y_test, y_proba_mle, ax=ax)
        ax.set_title("ROC — MLE")
        st.pyplot(fig, use_container_width=True)
        st.caption("**ROC**: AUC càng cao càng tốt; đo khả năng xếp hạng đúng xác suất.")

    with colB:
        fig_pr, ax_pr = make_fig(4.0, 3.0)
        PrecisionRecallDisplay.from_predictions(y_test, y_proba_mle, ax=ax_pr)
        ax_pr.set_title("PR — MLE")
        st.pyplot(fig_pr, use_container_width=True)
        st.caption("**PR**: hữu ích khi dữ liệu lệch lớp; trade-off giữa Precision và Recall.")

    with colC:
        if _HAS_CALIB:
            fig_cal, ax_cal = make_fig(4.0, 3.0)
            CalibrationDisplay.from_predictions(y_test, y_proba_mle, n_bins=10, strategy="uniform", ax=ax_cal)
            ax_cal.set_title("Calibration — MLE")
            st.pyplot(fig_cal, use_container_width=True)
            st.caption("**Calibration**: đường gần chéo → xác suất dự đoán phản ánh tần suất thực.")
        else:
            st.info("Calibration plot requires newer scikit-learn.")

# Coefficients — MLE (Top 10 by |coef|)
try:
    preproc = mle_pipe.named_steps["preprocess"]
    clf = mle_pipe.named_steps["clf"]
    ohe = preproc.named_transformers_["cat"].named_steps["onehot"] if len(categorical_cols) else None
    cat_names = list(ohe.get_feature_names_out(categorical_cols)) if ohe is not None else []
    feature_names = list(numeric_cols) + cat_names
    coefs = clf.coef_.ravel()
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    top10_mle = (coef_df.assign(abs_coef=lambda d: d["coef"].abs())
                         .sort_values("abs_coef", ascending=False)
                         .head(10)
                         .sort_values("coef"))
    fig_b_mle, ax_b_mle = make_fig(5.2, max(2.6, len(top10_mle)/12))
    ax_b_mle.barh(top10_mle["feature"], top10_mle["coef"])
    ax_b_mle.set_title("MLE Coefficients (Top 10 by |coef|)")
    st.pyplot(fig_b_mle, use_container_width=True)
    st.caption("**Diễn giải**: Hệ số dương → tăng xác suất lớp 1 khi biến tăng (giữ các biến khác cố định). Lưu ý: input đã chuẩn hoá & one-hot, hệ số phụ thuộc baseline của one-hot.")
except Exception:
    st.info("Không lấy được hệ số MLE.")

# ========= Bayesian =========
st.header("🧮 Bayesian Logistic Regression")

if bayes_method == "Laplace (nhanh)":
    C_map = prior_sigma_w**2
    map_clf = LogisticRegression(penalty="l2", C=C_map, solver="lbfgs", max_iter=1000, class_weight=class_weight)
    map_pipe = Pipeline([("preprocess", preprocess), ("clf", map_clf)])
    map_pipe.fit(X_train, y_train)
    y_pred_map = map_pipe.predict(X_test)
    y_proba_map = map_pipe.predict_proba(X_test)[:, 1]

    map_mets = summarize_metrics(y_test, y_pred_map, y_proba_map)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Accuracy (MAP)", f"{map_mets['Accuracy']:.3f}")
    with c2: st.metric("Precision (MAP)", f"{map_mets['Precision']:.3f}")
    with c3: st.metric("Recall (MAP)", f"{map_mets['Recall']:.3f}")
    with c4: st.metric("F1 (MAP)", f"{map_mets['F1']:.3f}")
    st.metric("ROC AUC (MAP)", f"{map_mets.get('ROC AUC', float('nan')):.3f}")
    st.caption("**MAP ≈ Logistic L2** với **C≈σ²** (prior Gaussian).")

    cm_map = confusion_matrix(y_test, y_pred_map)
    st.write("**Confusion Matrix (MAP)**")
    st.write(pd.DataFrame(cm_map, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

    rowA, rowB, rowC = st.columns(3)
    with rowA:
        fig2, ax2 = make_fig(4.0, 3.0)
        RocCurveDisplay.from_predictions(y_test, y_proba_map, ax=ax2)
        ax2.set_title("ROC — MAP")
        st.pyplot(fig2, use_container_width=True)
    with rowB:
        fig_pr2, ax_pr2 = make_fig(4.0, 3.0)
        PrecisionRecallDisplay.from_predictions(y_test, y_proba_map, ax=ax_pr2)
        ax_pr2.set_title("PR — MAP")
        st.pyplot(fig_pr2, use_container_width=True)
    with rowC:
        if _HAS_CALIB:
            fig_cal2, ax_cal2 = make_fig(4.0, 3.0)
            CalibrationDisplay.from_predictions(y_test, y_proba_map, n_bins=10, strategy="uniform", ax=ax_cal2)
            ax_cal2.set_title("Calibration — MAP")
            st.pyplot(fig_cal2, use_container_width=True)
        else:
            st.info("Calibration plot requires newer scikit-learn.")

    # Laplace covariance & coefficient uncertainty
    Z_train = build_design_matrix(preprocess, X_train)
    w_map = map_pipe.named_steps["clf"].coef_.ravel()
    b_map = float(map_pipe.named_steps["clf"].intercept_[0])
    Sigma = laplace_posterior(
        Z_train, w_map, b_map, prior_var_w=prior_sigma_w**2, prior_var_b=prior_sigma_b**2
    )

    diag_Sigma = np.diag(Sigma)
    st.caption(f"**Laplace posterior Σ**: shape {Sigma.shape}, trace ≈ {diag_Sigma.sum():.3f} (bất định tổng quát).")

    try:
        ohe_map = preprocess.named_transformers_["cat"].named_steps["onehot"] if len(categorical_cols) else None
        cat_names_map = list(ohe_map.get_feature_names_out(categorical_cols)) if ohe_map is not None else []
        feature_names_map = list(numeric_cols) + cat_names_map
        se = np.sqrt(diag_Sigma[1:1+len(feature_names_map)])
        coef_df_map = pd.DataFrame({"feature": feature_names_map, "coef": w_map, "se": se})
        top10 = (coef_df_map.assign(abs_coef=lambda d: d["coef"].abs())
                              .sort_values("abs_coef", ascending=False)
                              .head(10)
                              .sort_values("coef"))
        fig_b_map, ax_b_map = make_fig(5.2, max(2.6, len(top10)/12))
        ax_b_map.barh(top10["feature"], top10["coef"], xerr=1.96*top10["se"])
        ax_b_map.set_title("MAP Coefficients (Top 10 by |coef|) ±1.96·SE")
        st.pyplot(fig_b_map, use_container_width=True)
        st.caption("**Thanh lỗi** ≈ độ bất định từ Laplace. Thanh dài → bất định cao.")
    except Exception:
        st.info("Không vẽ được sai số hệ số (Laplace).")

    # Save for Predict UI
    st.session_state["predict_mode"] = "laplace"
    st.session_state["preprocess"] = preprocess
    st.session_state["mle_pipe"] = mle_pipe
    st.session_state["map_pipe"] = map_pipe
    st.session_state["Sigma"] = Sigma
    st.session_state["w_map"] = w_map
    st.session_state["b_map"] = b_map
    st.session_state["feature_names_all"] = (numeric_cols, categorical_cols)

elif bayes_method == "PyMC sampling":
    if not _PYMC_AVAILABLE:
        st.error("PyMC/ArviZ chưa sẵn sàng trong môi trường này.")
    else:
        # Build design matrices
        Xt_train = build_design_matrix(preprocess, X_train)
        Xt_test  = build_design_matrix(preprocess, X_test)
        try:
            with st.spinner("Đang sampling với PyMC (cần compiler 64-bit)..."):
                with pm.Model() as bayes_logit:
                    beta = pm.Normal("beta", mu=0, sigma=prior_sigma_beta_pymc, shape=Xt_train.shape[1])
                    intercept = pm.Normal("intercept", mu=0, sigma=prior_sigma_intercept_pymc)
                    logits = intercept + pm.math.dot(Xt_train, beta)
                    p = pm.math.sigmoid(logits)
                    y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train.values)
                    idata = pm.sample(
                        draws=int(draws), tune=int(tune),
                        target_accept=float(target_accept),
                        chains=int(chains), cores=int(cores),
                        random_seed=42, progressbar=True
                    )
            st.success("✅ PyMC fitted.")
            try:
                summ = az.summary(idata, var_names=["beta","intercept"], round_to=3)
                st.dataframe(summ.head(12))
            except Exception:
                st.info("Không thể hiển thị bảng summary ArviZ.")

            beta_mean = idata.posterior["beta"].mean(dim=("chain","draw")).values
            intercept_mean = idata.posterior["intercept"].mean(dim=("chain","draw")).values
            logits_test = intercept_mean + Xt_test @ beta_mean
            proba_bayes = 1/(1+np.exp(-logits_test))
            y_pred_bayes = (proba_bayes >= 0.5).astype(int)

            acc_b  = accuracy_score(y_test, y_pred_bayes)
            prec_b = precision_score(y_test, y_pred_bayes, zero_division=0)
            rec_b  = recall_score(y_test, y_pred_bayes, zero_division=0)
            f1_b   = f1_score(y_test, y_pred_bayes, zero_division=0)
            auc_b  = roc_auc_score(y_test, proba_bayes)

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Accuracy (Bayes)", f"{acc_b:.3f}")
            with c2: st.metric("Precision (Bayes)", f"{prec_b:.3f}")
            with c3: st.metric("Recall (Bayes)", f"{rec_b:.3f}")
            with c4: st.metric("F1 (Bayes)", f"{f1_b:.3f}")
            st.metric("ROC AUC (Bayes)", f"{auc_b:.3f}")
            st.caption("PyMC trả về posterior đầy đủ → có thể trích xuất CI & posterior predictive.")

            st.session_state["predict_mode"] = "pymc"
            st.session_state["preprocess"] = preprocess
            st.session_state["mle_pipe"] = mle_pipe
            st.session_state["idata"] = idata
            st.session_state["Xt_train_shape"] = Xt_train.shape
            st.session_state["feature_names_all"] = (numeric_cols, categorical_cols)

        except Exception as e:
            st.error("PyMC sampling failed.\n\n````\n%s\n````" % (str(e)[:1000]))
            st.info("▶️ Auto-fallback to Bayesian (Laplace).")
            C_map = 100.0
            map_clf = LogisticRegression(penalty="l2", C=C_map, solver="lbfgs", max_iter=1000, class_weight=class_weight)
            map_pipe = Pipeline([("preprocess", preprocess), ("clf", map_clf)])
            map_pipe.fit(X_train, y_train)

            Z_train = build_design_matrix(preprocess, X_train)
            w_map = map_pipe.named_steps["clf"].coef_.ravel()
            b_map = float(map_pipe.named_steps["clf"].intercept_[0])
            Sigma = laplace_posterior(Z_train, w_map, b_map, prior_var_w=100.0, prior_var_b=100.0)

            st.session_state["predict_mode"] = "laplace"
            st.session_state["preprocess"] = preprocess
            st.session_state["mle_pipe"] = mle_pipe
            st.session_state["map_pipe"] = map_pipe
            st.session_state["Sigma"] = Sigma
            st.session_state["w_map"] = w_map
            st.session_state["b_map"] = b_map
            st.session_state["feature_names_all"] = (numeric_cols, categorical_cols)

# ========= Model Comparison =========
if "map_pipe" in st.session_state or ("idata" in st.session_state and _PYMC_AVAILABLE):
    st.header("⚖️ Model Comparison (MLE vs Bayesian)")
    comp = []
    def safe_mets(name, y_true, proba, y_pred):
        d = {"Model": name,
             "Accuracy": accuracy_score(y_true, y_pred),
             "Precision": precision_score(y_true, y_pred, zero_division=0),
             "Recall": recall_score(y_true, y_pred, zero_division=0),
             "F1": f1_score(y_true, y_pred, zero_division=0)}
        try:
            d["ROC AUC"] = roc_auc_score(y_true, proba)
        except Exception:
            d["ROC AUC"] = np.nan
        return d

    # MLE
    comp.append(safe_mets("MLE", y_test, y_proba_mle, y_pred_mle))

    # Bayesian
    if "map_pipe" in st.session_state:
        p_map = st.session_state["map_pipe"].predict_proba(X_test)[:,1]
        comp.append(safe_mets("Bayesian (MAP/Laplace)", y_test, p_map, (p_map>=0.5).astype(int)))
    elif "idata" in st.session_state and _PYMC_AVAILABLE:
        Xt_test = build_design_matrix(st.session_state["preprocess"], X_test)
        beta_mean = st.session_state["idata"].posterior["beta"].mean(dim=("chain","draw")).values
        intercept_mean = st.session_state["idata"].posterior["intercept"].mean(dim=("chain","draw")).values
        logits_test = intercept_mean + Xt_test @ beta_mean
        p_bayes = 1/(1+np.exp(-logits_test))
        comp.append(safe_mets("Bayesian (PyMC mean)", y_test, p_bayes, (p_bayes>=0.5).astype(int)))

    st.dataframe(pd.DataFrame(comp).round(3))
    st.caption("**Gợi ý**: Chọn mô hình theo mục tiêu (ví dụ Recall cao). Nếu chênh lệch nhỏ, ưu tiên mô hình đơn giản (MLE) để triển khai dễ dàng.")

# ========= Threshold & Gains/Lift =========
st.header("📏 Threshold & Gains / Lift")
# Choose probability source: prefer Bayesian MAP if available, else MLE
if "map_pipe" in st.session_state:
    proba = st.session_state["map_pipe"].predict_proba(X_test)[:,1]
    src_name = "MAP (Laplace)"
else:
    proba = y_proba_mle if y_proba_mle is not None else None
    src_name = "MLE"

if proba is None:
    st.info("Cần xác suất (proba) để phân tích ngưỡng. Hãy dùng solver hỗ trợ `predict_proba`.")
else:
    # Recommend threshold by maximizing F1
    cand = np.linspace(0.05, 0.95, 19)
    f1s = []
    for t in cand:
        y_t = (proba >= t).astype(int)
        f1s.append(f1_score(y_test, y_t, zero_division=0))
    t_rec = float(cand[int(np.argmax(f1s))])

    thr = st.slider(f"Decision threshold (using {src_name} probabilities)", 0.0, 1.0, t_rec, 0.01)
    y_thr = (proba >= thr).astype(int)
    mets = summarize_metrics(y_test, y_thr, proba)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Accuracy@thr", f"{mets['Accuracy']:.3f}")
    with c2: st.metric("Precision@thr", f"{mets['Precision']:.3f}")
    with c3: st.metric("Recall@thr", f"{mets['Recall']:.3f}")
    with c4: st.metric("F1@thr", f"{mets['F1']:.3f}")
    cm_thr = confusion_matrix(y_test, y_thr)
    st.write("**Confusion Matrix @ threshold**")
    st.write(pd.DataFrame(cm_thr, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))
    st.caption("**Hướng dẫn**: • Ưu tiên **Recall** → kéo ngưỡng ↓. • Ưu tiên **Precision** → kéo ngưỡng ↑. • F1 cân bằng cả hai.")

    # Lift by decile
    df_scores = pd.DataFrame({"y": y_test.values, "score": proba})
    df_scores = df_scores.sort_values("score", ascending=False).reset_index(drop=True)
    df_scores["decile"] = pd.qcut(df_scores["score"].rank(method="first"), 10, labels=False)
    lift = df_scores.groupby("decile")["y"].mean()
    base = df_scores["y"].mean()
    fig_lift, ax_lift = make_fig(5.2, 3.2)
    ax_lift.plot(lift.index, lift.values/base, marker="o")
    ax_lift.axhline(1.0, linestyle="--")
    ax_lift.set_title(f"Lift by Decile — {src_name}")
    ax_lift.set_xlabel("Decile (0 = top scored)")
    ax_lift.set_ylabel("Lift vs baseline")
    st.pyplot(fig_lift, use_container_width=True)
    st.caption("**Lift**: so sánh tỉ lệ dương tính trong mỗi decile với baseline (đường ngang=1). Decile đầu cao >>1 chứng tỏ mô hình xếp hạng tốt.")

# ========= Predict UI =========
st.header("🔮 Predict on Custom Input")
if "preprocess" not in st.session_state:
    st.info("Hãy chạy huấn luyện (MLE và một biến thể Bayesian) trước khi dự đoán.")
else:
    numeric_cols, categorical_cols = st.session_state["feature_names_all"]
    with st.form("predict_form"):
        input_data = {}
        for c in numeric_cols:
            try:
                col_min = float(pd.to_numeric(X_train[c], errors="coerce").min())
                col_max = float(pd.to_numeric(X_train[c], errors="coerce").max())
                col_mean = float(pd.to_numeric(X_train[c], errors="coerce").mean())
                step = (col_max-col_min)/100 if (col_max>col_min) else 1.0
                input_data[c] = st.number_input(f"{c} (numeric)", value=col_mean, min_value=col_min, max_value=col_max, step=step, help="Nhập số; đã được chuẩn hoá trong pipeline.")
            except Exception:
                input_data[c] = st.number_input(f"{c} (numeric)", value=0.0)
        for c in categorical_cols:
            opts = sorted([str(x) for x in pd.Series(X_train[c]).dropna().unique().tolist()])
            input_data[c] = st.selectbox(f"{c} (categorical)", options=opts if opts else [""], index=0, help="Chọn một trong các giá trị đã thấy ở train.")
        submitted = st.form_submit_button("Predict")

    if submitted:
        x_df = pd.DataFrame([input_data])
        mle_pipe = st.session_state["mle_pipe"]
        try:
            p_mle = float(mle_pipe.predict_proba(x_df)[0,1])
            y_mle = int(p_mle >= 0.5)
        except Exception:
            y_mle = int(mle_pipe.predict(x_df)[0])
            p_mle = np.nan

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("MLE")
            st.metric("Probability (class=1)", f"{p_mle:.3f}")
            st.metric("Label @ 0.5", "1" if y_mle==1 else "0")
            st.caption("Bạn có thể dùng ngưỡng khác trong phần **Threshold**.")

        mode = st.session_state.get("predict_mode", "laplace")
        if mode == "laplace":
            preprocess = st.session_state["preprocess"]
            z = build_design_matrix(preprocess, x_df)
            z_aug = np.hstack([np.ones((1,1)), z])
            Sigma = st.session_state["Sigma"]
            theta_mean = np.concatenate([[st.session_state["b_map"]], st.session_state["w_map"]])
            m = float(z_aug @ theta_mean)
            v = float(z_aug @ Sigma @ z_aug.T)
            p_bayes = bayes_pred_prob_with_variance_correction(m, v)
            y_bayes = int(p_bayes >= 0.5)
            with c2:
                st.subheader("Bayesian (Laplace)")
                st.metric("Probability (var-corrected)", f"{p_bayes:.3f}")
                st.metric("Label @ 0.5", str(y_bayes))
                st.caption(f"Mean m={m:.3f}, variance v={v:.6f}. Xác suất đã điều chỉnh theo bất định.")
        elif mode == "pymc" and _PYMC_AVAILABLE and ("idata" in st.session_state):
            preprocess = st.session_state["preprocess"]
            z = build_design_matrix(preprocess, x_df)  # (1,d)
            idata = st.session_state["idata"]
            beta_draws = idata.posterior["beta"].values  # (chain, draw, d)
            intercept_draws = idata.posterior["intercept"].values  # (chain, draw)
            c, d, D = beta_draws.shape[0], beta_draws.shape[1], beta_draws.shape[2]
            samples = beta_draws.reshape(c*d, D)
            intercepts = intercept_draws.reshape(c*d)
            logits = intercepts + samples @ z.ravel()
            probs = 1/(1+np.exp(-logits))
            mean_p = float(np.mean(probs))
            lo, hi = float(np.quantile(probs, 0.025)), float(np.quantile(probs, 0.975))
            y_bayes = int(mean_p >= 0.5)

            with c2:
                st.subheader("Bayesian (PyMC)")
                st.metric("Mean probability", f"{mean_p:.3f}")
                st.metric("Label @ 0.5", str(y_bayes))
                st.caption(f"95% credible interval for p: [{lo:.3f}, {hi:.3f}]")
        else:
            with c2:
                st.subheader("Bayesian")
                st.info("Chưa có kết quả Bayesian để dự đoán (hãy chạy Laplace hoặc PyMC).")

# ========= Download predictions =========
st.header("⬇️ Download Test Predictions")
out_df = X_test.copy()
out_df[target_col + "_true"] = y_test.values
out_df["pred_MLE"] = y_pred_mle
out_df["proba_MLE"] = y_proba_mle if y_proba_mle is not None else np.nan

if "map_pipe" in st.session_state:
    proba_map = st.session_state["map_pipe"].predict_proba(X_test)[:,1]
    pred_map = (proba_map >= 0.5).astype(int)
    out_df["pred_MAP"] = pred_map
    out_df["proba_MAP"] = proba_map
elif "idata" in st.session_state and _PYMC_AVAILABLE:
    Xt_test = build_design_matrix(st.session_state["preprocess"], X_test)
    beta_mean = st.session_state["idata"].posterior["beta"].mean(dim=("chain","draw")).values
    intercept_mean = st.session_state["idata"].posterior["intercept"].mean(dim=("chain","draw")).values
    logits_test = intercept_mean + Xt_test @ beta_mean
    proba_bayes = 1/(1+np.exp(-logits_test))
    pred_bayes = (proba_bayes >= 0.5).astype(int)
    out_df["pred_Bayes"] = pred_bayes
    out_df["proba_Bayes"] = proba_bayes

csv_bytes = out_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_bytes, file_name="predictions_test.csv", mime="text/csv")

# ========= Playbooks & Glossary =========
with st.expander("🧭 Tuning Playbooks (What to do when...)", expanded=False):
    st.markdown("""
- **Recall quá thấp (bỏ sót nhiều bản ghi lớp 1)**  
  → Bật **class_weight='balanced'**, **giảm ngưỡng** trong phần Threshold, hoặc thu thập thêm dữ liệu lớp 1.
- **Precision quá thấp (nhiều false positive)**  
  → **Tăng ngưỡng**; cân nhắc **C nhỏ hơn** (L2 mạnh hơn) để làm xác suất bớt cực trị.
- **AUC/ROC thấp**  
  → Xem lại feature engineering; thêm biến, xử lý outlier, hoặc thử mô hình khác.
- **Calibration lệch (đường cong không gần chéo)**  
  → Tăng dữ liệu, thử regularization mạnh hơn, hoặc hậu hiệu chỉnh (Platt/Isotonic).
- **Hệ số khó diễn giải**  
  → Xem **Correlation**, hiểu baseline one-hot, hoặc báo cáo theo **mức tác động** (odds ratio = exp(coef)).
""")

with st.expander("📖 Glossary", expanded=False):
    st.markdown("""
- **Accuracy**: (TP+TN)/(all). Dễ “đẹp” khi lệch lớp nặng → cẩn thận.
- **Precision**: TP/(TP+FP). Trong các dự đoán “1”, bao nhiêu % đúng?
- **Recall (TPR)**: TP/(TP+FN). Trong các bản ghi “1” thật, bắt được bao nhiêu %?
- **F1**: Hài hoà Precision & Recall (thích hợp khi lệch lớp).
- **ROC AUC**: Khả năng xếp hạng đúng xác suất.
- **Lift**: Tỉ lệ dương tính trong nhóm top-k so với baseline → đánh giá khả năng ưu tiên.
- **Calibration**: Sự phù hợp giữa xác suất dự đoán và tần suất thực.
- **MAP/Laplace**: Ước lượng điểm cực đại hậu nghiệm + xấp xỉ Gaussian quanh nó (nhanh).
- **PyMC**: Lấy mẫu đầy đủ posterior (chậm hơn nhưng giàu thông tin).
""")

st.markdown("---")
st.caption("Plots are scaled via the sidebar slider. Each plot and section includes guidance to help you interpret and tune the model.")
