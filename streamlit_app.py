# app.py — OASIS Longitudinal (NaN-safe, no upload)
# Streamlit Dashboard: EDA, Training, Single-Case Prediction

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st

from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="OASIS – Alzheimer’s ML Dashboard", layout="wide", page_icon="🧠")

# ---------------------- Config ----------------------
DEFAULT_PATH = "oasis_longitudinal.csv"  # fixed OASIS dataset path
PRIORITY_LABELS = ["group"]  # prefer 'Group'
ID_LIKE_HINTS = ["subject id", "mri id"]  # typical OASIS ID-ish columns

# ---------------------- Utilities ----------------------
def pick_label_column(df: pd.DataFrame) -> str:
    # Prefer 'Group' if present and sensible
    if "Group" in df.columns and df["Group"].nunique(dropna=True) > 1:
        return "Group"
    # Otherwise use heuristics
    cols = list(df.columns)
    lowermap = {c: c.lower() for c in cols}
    for c in cols:
        if lowermap[c] in PRIORITY_LABELS and df[c].nunique(dropna=True) > 1:
            return c
    candidates = []
    for c in cols:
        nunq = df[c].nunique(dropna=True)
        if 1 < nunq <= 10 and nunq < len(df) * 0.5:
            candidates.append((nunq, c))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    return cols[-1]

def suggest_drop_cols(df: pd.DataFrame) -> List[str]:
    drops = []
    for c in df.columns:
        if c.lower() in ID_LIKE_HINTS:
            drops.append(c)
    # Nearly-unique identifiers
    n = len(df)
    for c in df.columns:
        if df[c].nunique(dropna=True) > 0.9 * n and c not in drops and c != "Group":
            drops.append(c)
    # Keep 'Visit' if present
    drops = [d for d in drops if d.lower() != "visit"]
    return sorted(list(set(drops)))

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not os.path.exists(DEFAULT_PATH):
        st.error(
            f"Data file not found at `{DEFAULT_PATH}`.\n\n"
            "Please place the OASIS CSV there or change DEFAULT_PATH in the code."
        )
        st.stop()
    df = pd.read_csv(DEFAULT_PATH)
    return df

def split_features_target(df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    cols = [c for c in df.columns if c != target_col and c not in drop_cols]
    X = df[cols].copy()
    y = df[target_col].copy()
    return X, y

def type_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "category" or X[c].dtype == "bool"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols

def make_pipeline(num_cols: List[str], cat_cols: List[str], model_name: str, params: Dict):
    # Robust preprocessing: imputation + scaling/OHE
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ]
    )
    if model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=params.get("max_iter", 300), C=params.get("C", 1.0))
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", None),
            random_state=42
        )
    elif model_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 3),
            random_state=42
        )
    else:
        raise ValueError("Unknown model")
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe

def safe_roc_auc(model, X_val, y_val):
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_val)
            if proba.shape[1] > 2:
                return roc_auc_score(y_val, proba, multi_class="ovr")
            else:
                return roc_auc_score(y_val, proba[:, 1])
        return np.nan
    except Exception:
        return np.nan

@st.cache_resource(show_spinner=False)
def train_model(X, y, model_name, params, test_size, random_state):
    # Guard: stratify needs >=2 classes
    if pd.Series(y).nunique(dropna=True) < 2:
        raise ValueError("Target has fewer than 2 classes after cleaning. Please adjust target or filters.")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    num_cols, cat_cols = type_columns(X_train)
    pipe = make_pipeline(num_cols, cat_cols, model_name, params)
    pipe.fit(X_train, y_train)
    yhat = pipe.predict(X_val)
    metrics = {
        "accuracy": accuracy_score(y_val, yhat),
        "f1_macro": f1_score(y_val, yhat, average="macro"),
        "roc_auc": safe_roc_auc(pipe, X_val, y_val),
        "report": classification_report(y_val, yhat, digits=3)
    }
    cm = confusion_matrix(y_val, yhat, labels=np.unique(y))
    labels = list(np.unique(y))
    return pipe, metrics, cm, labels, (X_train, X_val, y_train, y_val)

def auto_build_single_input(df: pd.DataFrame, X_cols: List[str]) -> Dict[str, any]:
    st.subheader("Enter values")
    values = {}
    for c in X_cols:
        series = df[c]
        if series.dtype == "object" or str(series.dtype) == "category" or series.dtype == "bool":
            opts = sorted([o for o in series.dropna().unique().tolist()])
            values[c] = st.selectbox(c, options=opts if len(opts) > 0 else [""])
        else:
            s = pd.to_numeric(series, errors="coerce").dropna()
            if len(s) == 0:
                values[c] = st.number_input(c, value=0.0)
            else:
                mn, mx = float(s.min()), float(s.max())
                mean_val = float(s.mean())
                step = (mx - mn) / 100 if mx > mn else 1.0
                if np.isfinite(mn) and np.isfinite(mx):
                    values[c] = st.number_input(c, value=mean_val, min_value=mn, max_value=mx, step=step)
                else:
                    values[c] = st.number_input(c, value=mean_val if np.isfinite(mean_val) else 0.0)
    return values

# ---------------------- Sidebar / Navigation ----------------------
st.sidebar.title("🧠 OASIS – Alzheimer’s ML Dashboard")
st.sidebar.markdown(f"**Data path:** `{DEFAULT_PATH}`")

# Load raw df and select label first
raw_df = load_data()
auto_label = pick_label_column(raw_df)
label_col = st.sidebar.selectbox(
    "Target column",
    options=list(raw_df.columns),
    index=list(raw_df.columns).index(auto_label) if auto_label in raw_df.columns else 0
)
st.sidebar.caption(f"Auto-detected: **{auto_label}**")

# ---- Clean missing values AFTER selecting target ----
df = raw_df.replace([np.inf, -np.inf], np.nan)

if label_col not in df.columns:
    st.error("Selected target column not in dataframe.")
    st.stop()

missing_target_rows = df[label_col].isna().sum()
if missing_target_rows > 0:
    st.warning(f"Dropping {missing_target_rows} rows with missing target `{label_col}`.")
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

# Drop all-empty feature cols (entirely NaN)
empty_feature_cols = [c for c in df.columns if c != label_col and df[c].isna().all()]
if empty_feature_cols:
    st.info(f"Dropping all-empty columns: {', '.join(empty_feature_cols)}")
    df = df.drop(columns=empty_feature_cols)

# Now compute drop suggestions on the cleaned df
default_drops = [c for c in ["Subject ID", "MRI ID"] if c in df.columns]
drop_suggest = sorted(list(set(default_drops + suggest_drop_cols(df))))
drop_cols = st.sidebar.multiselect(
    "Columns to drop (IDs, nearly-unique)",
    options=list(df.columns),
    default=[c for c in drop_suggest if c != label_col]
)

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Overview", "Explore (EDA)", "Train & Evaluate", "Single-Case Prediction"])

# Derive X/y
if label_col not in df.columns:
    st.error("Selected target column not in dataframe after cleaning.")
    st.stop()
X, y = split_features_target(df, label_col, drop_cols)

# ---------------------- Pages ----------------------
if page == "Overview":
    st.title("Overview")
    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        st.markdown("### Dataset Snapshot")
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown(f"- **Rows:** {len(df)}  \n- **Columns:** {len(df.columns)}")
        st.markdown(f"- **Target:** `{label_col}`  \n- **Dropped:** {', '.join(drop_cols) if drop_cols else 'None'}")

    with c2:
        st.markdown("### Class Distribution")
        vc = y.value_counts(dropna=False)
        st.write(vc)
        cd = pd.DataFrame({"class": vc.index.astype(str), "count": vc.values})
        chart = (
            alt.Chart(cd)
            .mark_bar()
            .encode(x=alt.X("class:N", sort="-y", title="Class"),
                    y=alt.Y("count:Q", title="Count"),
                    tooltip=["class", "count"])
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Column Types")
    num_cols, cat_cols = type_columns(X)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Numeric:**", num_cols if num_cols else "—")
    with col2:
        st.write("**Categorical:**", cat_cols if cat_cols else "—")

elif page == "Explore (EDA)":
    st.title("Explore (EDA)")
    st.markdown("Select columns to visualize distributions and relationships.")

    if len(X.columns) == 0:
        st.error("No feature columns available after drops. Please adjust drop list.")
    else:
        col = st.selectbox("Choose a column", options=list(X.columns))
        if col:
            if X[col].dtype == "object" or str(X[col].dtype) == "category" or X[col].dtype == "bool":
                st.markdown(f"#### Categorical: {col}")
                vc = X[col].value_counts(dropna=False)
                cat_df = pd.DataFrame({col: vc.index.astype(str), "count": vc.values})
                chart = (
                    alt.Chart(cat_df)
                    .mark_bar()
                    .encode(x=alt.X(f"{col}:N", sort="-y"), y="count:Q", tooltip=[col, "count"])
                    .properties(height=350)
                )
                st.altair_chart(chart, use_container_width=True)
                st.markdown("#### Cross by Target")
                cross = pd.crosstab(X[col].astype(str), y)
                st.dataframe(cross, use_container_width=True)
            else:
                st.markdown(f"#### Numeric: {col}")
                c1, c2 = st.columns(2)
                with c1:
                    hist = alt.Chart(pd.DataFrame({col: X[col]})).mark_bar().encode(
                        alt.X(f"{col}:Q", bin=alt.Bin(maxbins=30)),
                        y="count()"
                    ).properties(height=350)
                    st.altair_chart(hist, use_container_width=True)
                with c2:
                    box = alt.Chart(pd.DataFrame({col: X[col], "target": y})).mark_boxplot().encode(
                        x="target:N", y=f"{col}:Q"
                    ).properties(height=350)
                    st.altair_chart(box, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Correlation (numeric only)")
    num_cols, _ = type_columns(X)
    if len(num_cols) >= 2:
        corr = X[num_cols].corr(numeric_only=True)
        st.dataframe(corr.style.background_gradient(cmap="Blues"), use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

elif page == "Train & Evaluate":
    st.title("Train & Evaluate")
    st.markdown("Tune settings, train a model, and review performance.")

    with st.form("train_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            model_name = st.selectbox("Model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
            test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.01)
        with c2:
            random_state = st.number_input("Random state", min_value=0, value=42, step=1)
        with c3:
            st.caption("Hyperparameters")
            params = {}
            if model_name == "Logistic Regression":
                params["C"] = st.number_input("C (inverse regularization)", min_value=0.001, value=1.0, step=0.1)
                params["max_iter"] = st.number_input("max_iter", min_value=50, value=300, step=50)
            elif model_name == "Random Forest":
                params["n_estimators"] = st.number_input("n_estimators", min_value=50, value=400, step=50)
                max_depth = st.text_input("max_depth (blank = None)", value="")
                params["max_depth"] = None if max_depth.strip() == "" else int(max_depth)
            elif model_name == "Gradient Boosting":
                params["n_estimators"] = st.number_input("n_estimators", min_value=50, value=300, step=50)
                params["learning_rate"] = st.number_input("learning_rate", min_value=0.001, value=0.05, step=0.01)
                params["max_depth"] = st.number_input("max_depth (trees)", min_value=1, value=3, step=1)

        submitted = st.form_submit_button("Train")
    if submitted:
        try:
            with st.spinner("Training..."):
                model, metrics, cm, labels, splits = train_model(X, y, model_name, params, test_size, random_state)

            st.success("Training complete.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            m2.metric("F1 (macro)", f"{metrics['f1_macro']:.3f}")
            m3.metric("ROC AUC", f"{metrics['roc_auc']:.3f}" if not np.isnan(metrics['roc_auc']) else "—")

            st.markdown("#### Confusion Matrix")
            fig, ax = plt.subplots()
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            st.pyplot(fig, clear_figure=True)

            st.markdown("#### Classification Report")
            st.code(metrics["report"])

            from sklearn.preprocessing import label_binarize
            X_train, X_val, y_train, y_val = splits
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_val)
                    fig2, ax2 = plt.subplots()
                    if proba.shape[1] == 2:
                        RocCurveDisplay.from_predictions(y_val, proba[:, 1], name="ROC", ax=ax2)
                    else:
                        classes = model.classes_
                        y_bin = label_binarize(y_val, classes=classes)
                        for i, cls in enumerate(classes):
                            RocCurveDisplay.from_predictions(y_bin[:, i], proba[:, i], name=f"ROC: {cls}", ax=ax2)
                    st.pyplot(fig2, clear_figure=True)
                except Exception:
                    st.info("Could not render ROC curves for this configuration.")

            # Persist for Single-Case Prediction
            st.session_state["trained_model"] = model
            st.session_state["feature_columns"] = list(X.columns)
            st.session_state["train_df"] = pd.concat([X, y.rename(label_col)], axis=1)

        except ValueError as e:
            st.error(f"Training failed: {e}")

elif page == "Single-Case Prediction":
    st.title("Single-Case Prediction")
    if "trained_model" not in st.session_state or "feature_columns" not in st.session_state:
        st.warning("Please train a model first on the **Train & Evaluate** page.")
    else:
        model = st.session_state["trained_model"]
        feat_cols = st.session_state["feature_columns"]
        train_df = st.session_state.get("train_df", pd.concat([X, y.rename(label_col)], axis=1))

        with st.form("single_case"):
            inputs = auto_build_single_input(train_df, feat_cols)
            predict_btn = st.form_submit_button("Predict")
        if predict_btn:
            x_row = pd.DataFrame([inputs])[feat_cols]
            pred = model.predict(x_row)[0]
            st.success(f"**Predicted class:** {pred}")
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_row).flatten()
                prob_df = pd.DataFrame({"class": list(model.classes_), "probability": proba})
                prob_df["probability"] = prob_df["probability"].round(4)
                st.markdown("#### Class Probabilities")
                st.dataframe(prob_df, use_container_width=True)
