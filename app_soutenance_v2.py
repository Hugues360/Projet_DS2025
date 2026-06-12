# app_soutenance.py
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

import pydeck as pdk
from streamlit_plotly_events import plotly_events
import plotly.express as px

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(
    page_title="Soutenance - Compagnon Immobilier",
    page_icon="🏠",
    layout="wide"
)
sns.set_style("whitegrid")

# ==========================================================
# HELPERS
# ==========================================================
def to_series(y):
    if y is None:
        return None
    if isinstance(y, pd.DataFrame):
        return y.iloc[:, 0]
    return y

def euro(x):
    return f"{x:,.0f} €".replace(",", " ")

def compute_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

def icon_type(row):
    if "typedebien_lite_a" in row and row["typedebien_lite_a"] == 1:
        return "🏢"
    if "typedebien_lite_m" in row and row["typedebien_lite_m"] == 1:
        return "🏠"
    return "🏷️"

def error_band(pct):
    if pct < 10:
        return "green"
    elif pct < 20:
        return "orange"
    return "red"

# ==========================================================
# DATA
# ==========================================================

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

    xgb = XGBRegressor(objective="reg:squarederror", random_state=42)
    xgb.fit(X_train_scaled, y_train)
    models["XGBRegressor"] = xgb

    # Tuned
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


def build_predictions(models, X_test_scaled, y_test):
    pred_store = {}
    for name, model in models.items():
        p = model.predict(X_test_scaled)
        pred_store[name] = p
    pred_df = pd.DataFrame(pred_store, index=X_test_scaled.index)
    pred_df["y_true"] = y_test.values
    return pred_df

# ==========================================================
# LOAD
# ==========================================================
data = load_data()
raw = data["raw"]
X_train_scaled = data["X_train_scaled"]
X_test_scaled = data["X_test_scaled"]
X_train = data["X_train"]
X_test = data["X_test"]
y_train = to_series(data["y_train"])
y_test = to_series(data["y_test"])

st.title("🏠 Projet Compagnon Immobilier — Soutenance 20 min")

if any(x is None for x in [X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test]):
    st.error("Fichiers manquants. Vérifie les CSV exportés depuis le notebook.")
    st.stop()

# harmonisation index
for d in [X_train_scaled, X_test_scaled, X_train, X_test]:
    d.index = d.index.astype(str)
for s in [y_train, y_test]:
    s.index = s.index.astype(str)

models = train_models(X_train_scaled, y_train)
pred_df = build_predictions(models, X_test_scaled, y_test)

# ==========================================================
# SIDEBAR
# ==========================================================
section = st.sidebar.radio(
    "Plan",
    [
        "1. Contexte & données",
        "2. Méthodologie détaillée",
        "3. Résultats avancés",
        "4. Analyse d'erreurs multi-modèles",
        "5. SHAP global",
        "6. Limites & perspectives",
        "7. Prédiction unitaire avancée"
    ]
)

# ==========================================================
# SECTION 1
# ==========================================================
if section == "1. Contexte & données":
    st.header("1) Contexte & topologie des données")

    if raw is None:
        st.info("Dataset brut non disponible.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Nb annonces", f"{len(raw):,}".replace(",", " "))
        c2.metric("Nb variables", raw.shape[1])
        c3.metric("Période", "2019-2023")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(raw["prix_bien"], bins=60, kde=True, ax=ax)
            ax.set_title("Distribution du prix_bien")
            st.pyplot(fig)

        with col2:
            if "surface" in raw.columns:
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.histplot(raw["surface"].dropna(), bins=60, kde=True, ax=ax)
                ax.set_title("Distribution de la surface")
                st.pyplot(fig)

        quanti = [c for c in ["prix_bien", "surface", "nbpieces", "nbchambres", "etage", "nb_etages"] if c in raw.columns]
        if len(quanti) >= 2:
            st.subheader("Corrélations (variables quantitatives)")
            corr = raw[quanti].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        if "surface" in raw.columns:
            st.subheader("Regplot : surface vs prix")
            fig, ax = plt.subplots(figsize=(7, 4))
            sample = raw.dropna(subset=["surface", "prix_bien"]).sample(min(4000, len(raw)), random_state=42)
            sns.regplot(data=sample, x="surface", y="prix_bien", scatter_kws={"alpha":0.2, "s":10}, line_kws={"color":"red"}, ax=ax)
            st.pyplot(fig)

        quali = [c for c in ["typedebien_lite", "classeenergie", "ges"] if c in raw.columns]
        for q in quali:
            st.subheader(f"Prix vs variable qualitative : {q}")
            fig, ax = plt.subplots(figsize=(8, 4))
            tmp = raw[[q, "prix_bien"]].dropna().copy()
            top_cat = tmp[q].value_counts().head(10).index
            tmp = tmp[tmp[q].isin(top_cat)]
            sns.boxplot(data=tmp[tmp["prix_bien"] < 1_000_000], x=q, y="prix_bien", ax=ax)
            plt.xticks(rotation=25)
            st.pyplot(fig)

# ==========================================================
# SECTION 2
# ==========================================================
elif section == "2. Méthodologie détaillée":
    st.header("2) Méthodologie détaillée (avec extraits de code)")

    st.markdown("### 2.1 Nettoyage")
    st.code(
        """# suppression colonnes quasi vides / fuite cible
cols_to_drop = ["prix_m2_vente", "reference", ...]
df = df.drop(columns=cols_to_drop, errors="ignore")""",
        language="python"
    )

    st.markdown("### 2.2 Imputation custom KNN")
    st.code(
        """# voisinage par type de bien + zone + nb pièces
# KNN regressif / majoritaire selon variable
for col in cols_with_na:
    df[col] = custom_knn_impute(df, group_cols=["typedebien_lite","nbpieces"])""",
        language="python"
    )

    st.markdown("### 2.3 Outliers")
    st.code(
        """# filtrage quantiles
q_low, q_hi = df["surface"].quantile([0.01, 0.99])
df = df[(df["surface"] >= q_low) & (df["surface"] <= q_hi)]""",
        language="python"
    )

    st.markdown("### 2.4 Feature engineering")
    st.code(
        """# target encoding, parsing texte, OHE
df["annee"] = pd.to_datetime(df["date"]).dt.year
df = pd.get_dummies(df, columns=["typedebien_lite", "chauffage"], drop_first=False)""",
        language="python"
    )

    st.markdown("### 2.5 Standardisation + split")
    st.code(
        """X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)""",
        language="python"
    )

# ==========================================================
# SECTION 3
# ==========================================================
elif section == "3. Résultats avancés":
    st.header("3) Résultats avancés")

    rows = []
    for name in models.keys():
        y_pred = pred_df[name].values
        m = compute_metrics(y_test.values, y_pred)
        rows.append([name, m["MAE"], m["RMSE"], m["R2"]])

    res = pd.DataFrame(rows, columns=["Modèle", "MAE", "RMSE", "R2"]).sort_values("MAE")
    st.dataframe(res.style.format({"MAE":"{:,.0f}", "RMSE":"{:,.0f}", "R2":"{:.4f}"}))

    st.subheader("Que montrer de plus ?")
    st.markdown("- Benchmark global\n- Robustesse par segments (type/surface)\n- Distribution des erreurs\n- % de prédictions sous seuil d’erreur")

    # % seuils par modèle
    kpi_rows = []
    for name in models.keys():
        ae = (pred_df[name] - pred_df["y_true"]).abs()
        pe = ae / pred_df["y_true"].replace(0, np.nan) * 100
        kpi_rows.append([
            name,
            (pe < 10).mean() * 100,
            (pe < 20).mean() * 100,
            (pe >= 20).mean() * 100
        ])
    kpi_df = pd.DataFrame(kpi_rows, columns=["Modèle", "% <10%", "% <20%", "% >=20%"])
    st.dataframe(kpi_df.style.format({"% <10%":"{:.1f}", "% <20%":"{:.1f}", "% >=20%":"{:.1f}"}))

# ==========================================================
# SECTION 4
# ==========================================================
elif section == "4. Analyse d'erreurs multi-modèles":
    st.header("4) Analyse d’erreurs multi-modèles")

    focus_type = st.selectbox("Focus type de bien", ["Tous", "Appartements", "Maisons"])
    surface_focus = st.slider("Focus surface", 0, 300, (0, 300))

    idx = X_test.index
    mask = pd.Series(True, index=idx)

    if focus_type == "Appartements" and "typedebien_lite_a" in X_test.columns:
        mask &= X_test["typedebien_lite_a"] == 1
    elif focus_type == "Maisons" and "typedebien_lite_m" in X_test.columns:
        mask &= X_test["typedebien_lite_m"] == 1

    if "surface" in X_test.columns:
        mask &= X_test["surface"].between(surface_focus[0], surface_focus[1], inclusive="both")

    selected_idx = mask[mask].index
    if len(selected_idx) < 20:
        st.warning("Pas assez d'observations avec ces filtres.")
    else:
        # Distribution erreurs comparée tous modèles
        err_df = []
        for name in models.keys():
            e = (pred_df.loc[selected_idx, name] - pred_df.loc[selected_idx, "y_true"]).values
            err_df.append(pd.DataFrame({"erreur": e, "Modèle": name}))
        err_df = pd.concat(err_df, axis=0)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.kdeplot(data=err_df, x="erreur", hue="Modèle", common_norm=False, ax=ax)
        ax.set_title("Distribution des erreurs par modèle (focus filtré)")
        st.pyplot(fig)

        # MAE par modèle sur focus
        mae_rows = []
        for name in models.keys():
            y_t = pred_df.loc[selected_idx, "y_true"].values
            y_p = pred_df.loc[selected_idx, name].values
            mae_rows.append([name, mean_absolute_error(y_t, y_p)])
        mae_df = pd.DataFrame(mae_rows, columns=["Modèle", "MAE_focus"]).sort_values("MAE_focus")
        st.dataframe(mae_df.style.format({"MAE_focus":"{:,.0f}"}))

# ==========================================================
# SECTION 5
# ==========================================================
elif section == "5. SHAP global":
    st.header("5) Interprétabilité SHAP globale")

    model_name = st.selectbox("Modèle SHAP", ["RandomForestRegressor_tuned", "XGBRegressor_tuned"])
    model = models[model_name]

    sample_n = st.slider("Taille échantillon SHAP", 50, 400, 200, 50)
    X_sample = X_test_scaled.iloc[:sample_n]

    with st.spinner("Calcul SHAP..."):
        explainer = shap.Explainer(model, X_train_scaled.iloc[:1000])
        shap_values = explainer(X_sample)

    fig = plt.figure(figsize=(8, 4))
    shap.plots.bar(shap_values, max_display=15, show=False)
    st.pyplot(fig)

    fig = plt.figure(figsize=(8, 4))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    st.pyplot(fig)

# ==========================================================
# SECTION 6
# ==========================================================
elif section == "6. Limites & perspectives":
    st.header("6) Limites & perspectives")
    st.markdown("""
    **Limites**
    - Prix d’annonce vs prix de transaction
    - Données bruitées, hétérogènes
    - Sensibilité aux biens atypiques
    
    **Perspectives**
    - Split temporel strict
    - Ajout données exogènes
    - Monitoring en production
    """)

# ==========================================================
# SECTION 7
# ==========================================================
elif section == "7. Prédiction unitaire avancée":
    st.header("7) Prédiction unitaire avancée")

    # Choix modèle
    default_model = "RandomForestRegressor_tuned" if "RandomForestRegressor_tuned" in models else list(models.keys())[0]
    model_name = st.selectbox("Modèle", list(models.keys()), index=list(models.keys()).index(default_model))
    model = models[model_name]

    # Prépare table locale
    df_local = X_test.copy()
    df_local["y_true"] = y_test
    df_local["y_pred"] = pred_df[model_name]
    df_local["abs_error"] = (df_local["y_pred"] - df_local["y_true"]).abs()
    df_local["pct_error"] = df_local["abs_error"] / df_local["y_true"].replace(0, np.nan) * 100

    # Filtres type + surface + pièces
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        type_filter = st.selectbox("Type de bien", ["Tous", "Appartements", "Maisons"])
    with col_f2:
        if "surface" in df_local.columns:
            smin, smax = int(np.nanmin(df_local["surface"])), int(np.nanmax(df_local["surface"]))
            surface_range = st.slider("Surface", smin, smax, (smin, smax))
        else:
            surface_range = None
    with col_f3:
        if "nbpieces" in df_local.columns:
            pmin, pmax = int(np.nanmin(df_local["nbpieces"])), int(np.nanmax(df_local["nbpieces"]))
            pieces_range = st.slider("Nb pièces", pmin, pmax, (pmin, pmax))
        else:
            pieces_range = None

    mask = pd.Series(True, index=df_local.index)
    if type_filter == "Appartements" and "typedebien_lite_a" in df_local.columns:
        mask &= df_local["typedebien_lite_a"] == 1
    elif type_filter == "Maisons" and "typedebien_lite_m" in df_local.columns:
        mask &= df_local["typedebien_lite_m"] == 1

    if surface_range and "surface" in df_local.columns:
        mask &= df_local["surface"].between(surface_range[0], surface_range[1], inclusive="both")
    if pieces_range and "nbpieces" in df_local.columns:
        mask &= df_local["nbpieces"].between(pieces_range[0], pieces_range[1], inclusive="both")

    df_f = df_local[mask].copy()
    if len(df_f) == 0:
        st.warning("Aucune annonce avec ces filtres.")
        st.stop()

    # KPI distribution erreurs
    k1, k2, k3 = st.columns(3)
    k1.metric("% annonces <10% erreur", f"{(df_f['pct_error'] < 10).mean()*100:.1f}%")
    k2.metric("% annonces <20% erreur", f"{(df_f['pct_error'] < 20).mean()*100:.1f}%")
    k3.metric("% annonces >=20% erreur", f"{(df_f['pct_error'] >= 20).mean()*100:.1f}%")

    # Tri ID par écart
    order = st.radio("Tri par % écart absolu", ["Croissant", "Décroissant"], horizontal=True)
    asc = True if order == "Croissant" else False
    df_f = df_f.sort_values("pct_error", ascending=asc)

    # Label ID avec icône + erreur
    labels = []
    for idx_, row in df_f.head(3000).iterrows():
        ico = icon_type(row)
        labels.append(f"{ico} {idx_} | abs={euro(row['abs_error'])} | err={row['pct_error']:.1f}%")

    selected_label = st.selectbox("ID annonce (jeu test)", labels)
    selected_idx = selected_label.split("|")[0].strip().split(" ")[-1]

    row = df_f.loc[selected_idx]
    pred_price = row["y_pred"]
    real_price = row["y_true"]
    abs_error = row["abs_error"]
    pct_error = row["pct_error"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Prix prédit", euro(pred_price))
    c2.metric("Prix réel", euro(real_price))
    c3.metric("Écart", f"{euro(abs_error)} ({pct_error:.2f}%)")

    # Caractéristiques + carte voisins
    left, right = st.columns([1, 1.2])

    with left:
        st.subheader("Caractéristiques du bien sélectionné")
        st.dataframe(df_f.loc[[selected_idx]].drop(columns=["y_true", "y_pred", "abs_error", "pct_error"]).T.rename(columns={selected_idx: "valeur"}))

    with right:
        st.subheader("Carte : bien + 5 voisins les plus proches")
        if all(c in df_f.columns for c in ["mapCoordonneesLatitude", "mapCoordonneesLongitude"]):
            lat0 = df_f.loc[selected_idx, "mapCoordonneesLatitude"]
            lon0 = df_f.loc[selected_idx, "mapCoordonneesLongitude"]

            neigh = df_f.copy()
            neigh["dist2"] = (neigh["mapCoordonneesLatitude"] - lat0)**2 + (neigh["mapCoordonneesLongitude"] - lon0)**2
            neigh = neigh.sort_values("dist2").head(6).copy()  # selected + 5 voisins
            neigh["is_selected"] = neigh.index == selected_idx
            neigh["tooltip"] = neigh.apply(
                lambda r: f"ID={r.name}<br>Prix réel={euro(r['y_true'])}<br>Prix prédit={euro(r['y_pred'])}<br>Err={r['pct_error']:.1f}%",
                axis=1
            )

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=neigh.reset_index(),
                get_position="[mapCoordonneesLongitude, mapCoordonneesLatitude]",
                get_radius=120,
                get_fill_color="""
                    is_selected ? [0, 102, 255, 220] :
                    (pct_error < 10 ? [34,139,34,180] : (pct_error < 20 ? [255,165,0,180] : [220,20,60,180]))
                """,
                pickable=True,
            )

            view = pdk.ViewState(latitude=float(lat0), longitude=float(lon0), zoom=12)
            st.pydeck_chart(pdk.Deck(
                map_style=None,
                initial_view_state=view,
                layers=[layer],
                tooltip={"html": "{tooltip}", "style": {"backgroundColor": "steelblue", "color": "white"}}
            ))
        else:
            st.info("Colonnes mapCoordonneesLatitude/mapCoordonneesLongitude indisponibles.")

    # Carte globale erreurs colorées
    st.subheader("Carte globale des erreurs (vert/orange/rouge)")
    if all(c in df_f.columns for c in ["mapCoordonneesLatitude", "mapCoordonneesLongitude"]):
        map_df = df_f.copy().reset_index().rename(columns={"index": "idannonce"})
        map_df["color"] = map_df["pct_error"].apply(
            lambda x: [34, 139, 34, 170] if x < 10 else ([255, 165, 0, 170] if x < 20 else [220, 20, 60, 170])
        )
        map_df["tooltip"] = map_df.apply(
            lambda r: f"ID={r['idannonce']}<br>Réel={euro(r['y_true'])}<br>Prédit={euro(r['y_pred'])}<br>Err={r['pct_error']:.1f}%",
            axis=1
        )

        view = pdk.ViewState(latitude=float(map_df["mapCoordonneesLatitude"].mean()), longitude=float(map_df["mapCoordonneesLongitude"].mean()), zoom=10)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[mapCoordonneesLongitude, mapCoordonneesLatitude]",
            get_fill_color="color",
            get_radius=80,
            pickable=True
        )
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=view,
            layers=[layer],
            tooltip={"html": "{tooltip}", "style": {"backgroundColor": "#333", "color": "white"}}
        ))
    else:
        st.info("Carte indisponible : mapCoordonneesLatitude/mapCoordonneesLongitude manquantes.")

    # SHAP waterfall unitaire
    st.subheader("SHAP Waterfall (annonce sélectionnée)")
    with st.spinner("Calcul SHAP unitaire..."):
        explainer = shap.Explainer(model, X_train_scaled.iloc[:1000])
        x_one = X_test_scaled.loc[[selected_idx]]
        sv = explainer(x_one)

    fig = plt.figure(figsize=(9, 5))
    shap.plots.waterfall(sv[0], max_display=15, show=False)
    st.pyplot(fig)

st.markdown("---")
st.caption("Soutenance Data Science • Compagnon Immobilier")
