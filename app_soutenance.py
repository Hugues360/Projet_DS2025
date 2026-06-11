# app_soutenance.py
# ==========================================================
# Soutenance 20 min - Projet Compagnon Immobilier (Haut-Rhin)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import shap

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="Soutenance - Compagnon Immobilier",
    page_icon="🏠",
    layout="wide"
)
sns.set_style("whitegrid")

# -----------------------------
# Utils
# -----------------------------
def to_series(y):
    if y is None:
        return None
    if isinstance(y, pd.DataFrame):
        if "prix_bien" in y.columns:
            return y["prix_bien"]
        return y.iloc[:, 0]
    return y

def compute_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

def euro(x):
    return f"{x:,.0f} €".replace(",", " ")

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data():
    data = {}

    raw_path = Path("data/ech_annonces_ventes_68.csv")
    data["raw"] = pd.read_csv(raw_path, sep=";", index_col="idannonce") if raw_path.exists() else None

    files = {
        "X_train_scaled": "Notebook_X_train_scaled.csv",
        "X_test_scaled": "Notebook_X_test_scaled.csv",
        "X_train": "Notebook_X_train.csv",
        "X_test": "Notebook_X_test.csv",
        "y_train": "Notebook_y_train.csv",
        "y_test": "Notebook_y_test.csv",
    }

    for k, f in files.items():
        p = Path(f)
        data[k] = pd.read_csv(p, index_col="idannonce") if p.exists() else None

    return data

# -----------------------------
# Models
# -----------------------------
@st.cache_resource
def train_models(X_train_scaled, y_train):
    models = {}

    # Baselines
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    models["LinearRegression"] = lr

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    models["RandomForestRegressor"] = rf

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    )
    xgb.fit(X_train_scaled, y_train)
    models["XGBRegressor"] = xgb

    # Tuned RF (notebook)
    rf_tuned = RandomForestRegressor(
        n_estimators=1000,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.5,
        max_depth=None,
        criterion="squared_error",
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )
    rf_tuned.fit(X_train_scaled, y_train)
    models["RandomForestRegressor_tuned"] = rf_tuned

    # Tuned XGB (notebook)
    xgb_tuned = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42
    )
    xgb_tuned.fit(X_train_scaled, y_train)
    models["XGBRegressor_tuned"] = xgb_tuned

    return models

# -----------------------------
# App start
# -----------------------------
data = load_data()

raw = data["raw"]
X_train_scaled = data["X_train_scaled"]
X_test_scaled = data["X_test_scaled"]
X_train = data["X_train"]
X_test = data["X_test"]
y_train = to_series(data["y_train"])
y_test = to_series(data["y_test"])

st.title("🏠 Projet Compagnon Immobilier — Soutenance (20 min)")
st.markdown("**Prédiction du prix de vente des biens immobiliers (Haut-Rhin, 2019–2023)**")

if any(x is None for x in [X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test]):
    st.error("Certains fichiers CSV nécessaires sont manquants. Vérifie les exports du notebook.")
    st.stop()

# sécurité index: string pour cohérence selectbox
X_train_scaled.index = X_train_scaled.index.astype(str)
X_test_scaled.index = X_test_scaled.index.astype(str)
X_train.index = X_train.index.astype(str)
X_test.index = X_test.index.astype(str)
y_train.index = y_train.index.astype(str)
y_test.index = y_test.index.astype(str)

models = train_models(X_train_scaled, y_train)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Plan de soutenance")
section = st.sidebar.radio(
    "Aller à la section :",
    [
        "0. Executive summary (1 min)",
        "1. Contexte métier & données (3 min)",
        "2. Méthodologie ML (4 min)",
        "3. Résultats modèles (5 min)",
        "4. Analyse d'erreurs (3 min)",
        "5. Interprétabilité SHAP (3 min)",
        "6. Limites & perspectives (1 min)",
        "7. Prédiction unitaire (démo jury)"
    ]
)

# -----------------------------
# 0. Executive summary
# -----------------------------
if section == "0. Executive summary (1 min)":
    st.header("0) Executive summary")
    st.success(
        """
        **Objectif :** estimer le prix d’un bien à partir de ses caractéristiques.  
        **Résultat :** les modèles non linéaires (RandomForest / XGBoost) divisent l’erreur
        par ~2 vs régression linéaire.  
        **Message clé :** pipeline robuste + features métier = gain concret de précision.
        """
    )

# -----------------------------
# 1. Contexte & données
# -----------------------------
elif section == "1. Contexte métier & données (3 min)":
    st.header("1) Contexte métier & données")

    c1, c2, c3 = st.columns(3)
    c1.metric("Zone", "Haut-Rhin (68)")
    c2.metric("Période", "2019-2023")
    c3.metric("Cible", "prix_bien (€)")

    if raw is not None:
        st.write(f"**Volume brut :** {raw.shape[0]:,} annonces • {raw.shape[1]} variables".replace(",", " "))
        st.dataframe(raw.head(8))

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(raw["prix_bien"], kde=True, ax=ax)
            ax.set_title("Distribution de prix_bien")
            st.pyplot(fig)

        with col2:
            if "typedebien_lite" in raw.columns:
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.violinplot(
                    data=raw[raw["prix_bien"] < 1_000_000],
                    x="typedebien_lite",
                    y="prix_bien",
                    ax=ax
                )
                ax.set_title("Prix par type de bien (<1M€)")
                st.pyplot(fig)
    else:
        st.info("Fichier brut non trouvé (data/ech_annonces_ventes_68.csv).")

# -----------------------------
# 2. Méthodologie
# -----------------------------
elif section == "2. Méthodologie ML (4 min)":
    st.header("2) Méthodologie ML")
    st.markdown(
        """
        1. **Nettoyage des variables** (colonnes quasi vides, fuite cible `prix_m2_vente`, etc.)  
        2. **Imputation custom KNN** (type de bien + proximité géographique + nb pièces)  
        3. **Traitement des outliers** (quantiles sur variables pertinentes)  
        4. **Feature engineering**  
           - target encoding (commune/type/pièces/étages)  
           - année extraite de la date  
           - parsing exposition/chauffage/DPE/GES  
           - One-Hot Encoding final  
        5. **Standardisation**  
        6. **Modélisation** : LinearRegression, RandomForest, XGBoost (+ versions tuned)
        """
    )

# -----------------------------
# 3. Résultats modèles
# -----------------------------
elif section == "3. Résultats modèles (5 min)":
    st.header("3) Résultats des modèles")

    rows = []
    for name, model in models.items():
        pred = model.predict(X_test_scaled)
        m = compute_metrics(y_test, pred)
        rows.append([name, m["MAE"], m["RMSE"], m["R2"]])

    res = pd.DataFrame(rows, columns=["Modèle", "MAE", "RMSE", "R2"]).sort_values("MAE")
    st.dataframe(res.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R2": "{:.4f}"}))

    best_name = res.iloc[0]["Modèle"]
    st.success(f"🏆 Modèle retenu : **{best_name}**")

# -----------------------------
# 4. Analyse erreurs
# -----------------------------
elif section == "4. Analyse d'erreurs (3 min)":
    st.header("4) Analyse d'erreurs")

    model_name = st.selectbox("Modèle à analyser", list(models.keys()), index=list(models.keys()).index("RandomForestRegressor_tuned"))
    model = models[model_name]
    pred = model.predict(X_test_scaled)

    residuals = pred - y_test.values
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(residuals, bins=50, ax=ax)
    ax.set_title(f"Distribution des erreurs - {model_name} (prédit - réel)")
    st.pyplot(fig)

    if "surface" in X_test.columns:
        st.subheader("Performance par tranche de surface")
        bins = [0, 50, 100, 150, np.inf]
        labels = ["<=50", "50-100", "100-150", ">150"]
        cat = pd.cut(X_test["surface"], bins=bins, labels=labels)

        rows = []
        for lab in labels:
            mask = cat == lab
            if mask.sum() > 10:
                m = compute_metrics(y_test[mask], pred[mask])
                rows.append([lab, int(mask.sum()), m["MAE"], m["RMSE"], m["R2"]])

        seg_df = pd.DataFrame(rows, columns=["Surface", "Nb biens", "MAE", "RMSE", "R2"])
        st.dataframe(seg_df.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R2": "{:.3f}"}))

# -----------------------------
# 5. SHAP
# -----------------------------
elif section == "5. Interprétabilité SHAP (3 min)":
    st.header("5) Interprétabilité SHAP")

    model_name = st.selectbox(
        "Modèle SHAP",
        ["RandomForestRegressor_tuned", "XGBRegressor_tuned"],
        index=0
    )
    model = models[model_name]

    sample_n = st.slider("Taille échantillon SHAP", 50, 400, 200, 50)
    X_sample = X_test_scaled.iloc[:sample_n]

    with st.spinner("Calcul SHAP..."):
        explainer = shap.Explainer(model, X_train_scaled.iloc[:1000])
        shap_values = explainer(X_sample)

    st.subheader("Importance globale")
    fig = plt.figure(figsize=(8, 4))
    shap.plots.bar(shap_values, max_display=15, show=False)
    st.pyplot(fig)

    st.subheader("Distribution des effets")
    fig = plt.figure(figsize=(8, 4))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    st.pyplot(fig)

# -----------------------------
# 6. Limites
# -----------------------------
elif section == "6. Limites & perspectives (1 min)":
    st.header("6) Limites & perspectives")
    st.markdown(
        """
        **Limites**
        - Données d’annonces (prix affiché ≠ prix final signé)
        - Variabilité micro-locale forte
        - Cas atypiques (biens très grands / très haut de gamme)

        **Perspectives**
        - Validation temporelle stricte
        - Données externes (transport, revenus, taux)
        - Déploiement API + monitoring de dérive
        """
    )

# -----------------------------
# 7. Prédiction unitaire (demandée)
# -----------------------------
elif section == "7. Prédiction unitaire (démo jury)":
    st.header("7) Prédiction unitaire (démo jury)")
    st.markdown("Choisissez un modèle et un bien du test pour comparer **prix prédit** vs **prix réel**.")

    default_model = "RandomForestRegressor_tuned" if "RandomForestRegressor_tuned" in models else list(models.keys())[0]
    model_name = st.selectbox("Modèle", list(models.keys()), index=list(models.keys()).index(default_model))
    model = models[model_name]

    test_indices = X_test_scaled.index.tolist()
    selected_idx = st.selectbox("ID annonce (jeu test)", test_indices)

    x_row = X_test_scaled.loc[[selected_idx]]
    pred_price = float(model.predict(x_row)[0])

    real_price = float(y_test.loc[selected_idx])
    abs_error = abs(pred_price - real_price)
    rel_error = (abs_error / real_price * 100) if real_price != 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modèle", model_name)
    c2.metric("Prix prédit", euro(pred_price))
    c3.metric("Prix réel", euro(real_price))
    c4.metric("Écart absolu", euro(abs_error))

    if np.isnan(rel_error):
        st.write("**Erreur relative :** N/A")
    else:
        st.write(f"**Erreur relative :** {rel_error:.2f}%")

    if not np.isnan(rel_error):
        if rel_error < 10:
            st.success("✅ Prédiction très proche de la valeur réelle.")
        elif rel_error < 20:
            st.info("ℹ️ Prédiction correcte avec un écart modéré.")
        else:
            st.warning("⚠️ Écart plus important (cas potentiellement atypique).")

    st.subheader("Caractéristiques du bien sélectionné")
    if selected_idx in X_test.index:
        row_raw = X_test.loc[[selected_idx]].T.rename(columns={selected_idx: "valeur"})
        st.dataframe(row_raw)
    else:
        st.dataframe(x_row.T.rename(columns={selected_idx: "valeur (scaled)"}))

    with st.expander("Comparer tous les modèles sur ce bien"):
        comp = []
        for name, mdl in models.items():
            p = float(mdl.predict(x_row)[0])
            ae = abs(p - real_price)
            pe = (ae / real_price * 100) if real_price != 0 else np.nan
            comp.append([name, p, real_price, ae, pe])

        df_comp = pd.DataFrame(comp, columns=["Modèle", "Prix prédit", "Prix réel", "Erreur abs", "Erreur %"])
        df_comp = df_comp.sort_values("Erreur abs")
        st.dataframe(
            df_comp.style.format({
                "Prix prédit": "{:,.0f}",
                "Prix réel": "{:,.0f}",
                "Erreur abs": "{:,.0f}",
                "Erreur %": "{:.2f}"
            })
        )

st.markdown("---")
st.caption("Soutenance Data Science • Projet Compagnon Immobilier")
