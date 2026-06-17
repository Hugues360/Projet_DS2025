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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, SequentialFeatureSelector
import prince
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
    data["raw"] = pd.read_csv(raw_path, sep=";", index_col="idannonce")

    files = {
        "X_train_scaled": "Notebook_X_train_scaled.csv",
        "X_test_scaled": "Notebook_X_test_scaled.csv",
        "X_train": "Notebook_X_train.csv",
        "X_test": "Notebook_X_test.csv",
        "y_train": "Notebook_y_train.csv",
        "y_test": "Notebook_y_test.csv",
    }

    for k, f in files.items():
        data[k] = pd.read_csv(Path(f), index_col="idannonce")

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

st.title("🏠/🏢 Projet Compagnon Immobilier — Data Scientist #2 Orange")

if any(x is None for x in [X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test]):
    st.error("Fichiers manquants. Vérifie les CSV exportés depuis le notebook.")
    st.stop()

# harmonisation index
#for d in [X_train_scaled, X_test_scaled, X_train, X_test]:
#    d.index = d.index.astype(str)
#for s in [y_train, y_test]:
#    s.index = s.index.astype(str)

models = train_models(X_train_scaled, y_train)
pred_df = build_predictions(models, X_test_scaled, y_test)

# ==========================================================
# SIDEBAR
# ==========================================================
section = st.sidebar.radio(
    "Plan",
    [
        "1. Contexte & données",
        "2. Préprocessing & Feature engineering",
        "3. Méthodes de sélection de features",
        "4. Modélisation (LR, RFR & XGBR)",
        "5. Comparaisons des performances (Focus : surface et type de bien)",
        "6. Interprétabilité avec SHAP",
        "7. Limites & perspectives",
        "8. Démonstration intéractive"
    ]
)

# ==========================================================
# SECTION 1
# ==========================================================
if section == "1. Contexte & données":
    st.header("1) Contexte & données")

    st.success(
        """
        **Objectif :** Prédire le prix d’un bien à partir de ses caractéristiques.  
        **Données :** Repository Git ([klopstock-dviz/immo_vis](https://github.com/klopstock-dviz/immo_vis)).  
        **Périmètre :** 27K annonces immobilières de maisons et appartements dans le Haut-Rhin entre 2019 et 2023.  
        """
    )

    if raw is None:
        st.info("Dataset brut non disponible.")
    else:

        st.subheader("Pour ce projet nous nous basons sur le fichier data/ech_annonces_ventes_68.csv ([Github](https://raw.githubusercontent.com/klopstock-dviz/immo_vis/master/data/ech_annonces_ventes_68.csv))")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nb annonces", f"{len(raw):,}".replace(",", " "))
        c2.metric("Nb variables", raw.shape[1])
        c3.metric("Variable cible", "prix_bien")
        c4.metric("Période", "2019-2023")

        st.markdown("---")
        st.header("Découverte & exploration des données brutes :")
        
        with st.expander("50 premières lignes du Dataframe brut :"):
            st.dataframe(raw.head(50))

        c1, c2 = st.columns(2)
        c1.metric("Nombre de variable quantitative", len(raw.select_dtypes(include=["number","bool"]).columns.tolist()))
        c2.metric("Nombre de variable qualitative", len(raw.select_dtypes(include=["object"]).columns.tolist()))

        with st.expander("Top 20 des variables avec des valeurs manquantes"):
            na_ratio = (raw.isna().mean() * 100).sort_values(ascending=False).head(20)
            st.dataframe(na_ratio.to_frame("% NA"))

        with st.expander("Tableau descriptif des variables"):
            st.markdown(
                """
                | **Nom de la variable**             | **Description**                                                                                     |
                |-------------------------------------|-----------------------------------------------------------------------------------------------------|
                | **type_annonceur**                  | Catégorie : `'pr'` (PRO), `'pa'` (PARTICULIER).                                                    |
                | **typedebien**                      | Catégorie : `'m'` (Maison), `'a'` (Appartement), `'l'` (Lot), `'mn'` (Maison ?), `'an'` (Appartement ?). |
                | **typedetransaction**                | Catégorie : `'v'` (Vente), `'lt'`, `'vp'`, `'pi'` (à préciser).                                    |
                | **etage**                           | Numérique : étage du bien. -1 à 999 (médiane 0, moyenne 1.3).                                       |
                | **surface**                        | Surface en m² : 2 à 980 (médiane 95, moyenne 108.3).                                                 |
                | **surface_terrain**                | Surface du terrain en m² : 1 à 431 600 (médiane 589, moyenne 1017.4).                              |
                | **nb_pieces**                      | Nombre de pièces : 1 à 43 (médiane 4, moyenne 4.5).                                                   |
                | **prix_bien**                      | Prix en € : de 1^3 à 3.3^6 (médiane 2.4^5, moyenne 2.7^5).                                           |
                | **prix_maison**                    | Prix de la maison en € : 86 000 à 514 999 (médiane 199 620, moyenne 213 134.4). (peu de valeurs)   |
                | **prix_terrain**                   | Prix du terrain en € : 1 à 320 000 (médiane 87 000, moyenne 89 757.5). (peu de valeurs)            |
                | **mensualiteFinance**              | Remboursement mensuel : 0 à 7 715 (médiane 0, moyenne 13.2). Beaucoup de zéros.                   |
                | **balcon**                         | Nombre de balcons : 0 à 8 (médiane 0, moyenne 0.28). Beaucoup de valeurs nulles.                  |
                | **eau**                            | Nombre de salles d’eau : 0 à 10 (médiane 0, moyenne 0.28).                                           |
                | **bain**                           | Nombre de salles de bain : 0 à 12 (médiane 1, moyenne 0.68).                                         |
                | **dpeL**                          | Classe DPE : `'D'`, `'E'`, `'C'`, `'VI'`, `'0'`, `'B'`, `'F'`, `'NS'`, `'A'`, `'G'`.             |
                | **dpeC**                          | Score DPE : 0 à 988 (médiane 196, moyenne 205).                                                      |
                | **mapCoordonneesLatitude**        | Latitude GPS : 47.4 à 48.3. (Haut-Rhin)                                                               |
                | **mapCoordonneesLongitude**       | Longitude GPS : 6.85 à 7.61. (Haut-Rhin)                                                              |
                | **annonce_exclusive**             | Modalités : `'0'`, `'Oui'`, `'Non'`.                                                                |
                | **nb_etages**                     | Nombre d’étages : 0 à 36 (médiane 2, moyenne 2.87).                                                    |
                | **parking**                       | Variable vide (à vérifier).                                                                         |
                | **places_parking**                | Places de parking : 0 à 35 (médiane 2, moyenne 1.98).                                                  |
                | **cave**                          | Présence d’une cave : True / False.                                                                   |
                | **exposition**                    | Orientation : nombreuses modalités (ex. `'Sud'`, `'Nord'`, `'Est'`, `'Ouest'`, etc.), beaucoup de valeurs nulles. |
                | **ges_class**                     | Classe GES : `'D'`, `'E'`, `'C'`, `'B'`, `'A'`, `'F'`, `'VI'`, `'G'`, `'NS'`.                        |
                | **annee_construction**            | Année de construction : 971 à 2025 (médiane 1978).                                                     |
                | **nb_toilettes**                  | Nombre de toilettes : 0 à 15 (médiane 1).                                                              |
                | **nb_terraces**                   | Nombre de terrasses : 0 à 14 (médiane 1).                                                               |
                | **videophone**                     | Présence de videophone : True / False.                                                                  |
                | **porte_digicode**                 | Présence d’un digicode : True / False.                                                                   |
                | **surface_balcon**                 | Surface des balcons en m² : 1 à 801 (médiane 10, moyenne 12.48). Beaucoup de valeurs manquantes.     |
                | **ascenseur**                      | Présence d’un ascenseur : True / False. (beaucoup de valeurs manquantes)                              |
                | **nb_logements_copro**             | Nombre de logements en copropriété : 0 à 3442 (médiane 24). Beaucoup de valeurs manquantes.        |
                | **charges_copro**                   | Charges copro en € : 0 à 2.5^6 (médiane 1.2^3, moyenne 1.8^3). Beaucoup de valeurs manquantes.       |
                | **chauffage_energie**               | Type de chauffage : modalités variées, beaucoup de valeurs manquantes.                              |
                | **chauffage_systeme**                | Système de chauffage : modalités variées, beaucoup de valeurs manquantes.                         |
                | **chauffage_mode**                   | Mode de chauffage : modalités variées, beaucoup de valeurs manquantes.                            |
                | **categorie_annonceur**             | Catégorie d’annonceur : `'a'`, `'b'`, `'ca'`, `'cm'`, `'m'`, `'network'`. La majorité est `'a'`.   |
                | **logement_neuf**                  | Logement neuf : `'n'` (non), `'o'` (oui). 13% des ventes concernent des logements neufs.             |
                | **duree_int**                     | Variable à interpréter (distribution bimodale autour de -850 et 100).                              |
                | **typedebien_lite**                | Version simplifiée : `'a'` (Appartement), `'m'` (Maison), `'l'` (Lot).                              |
                | **date**                          | Date de publication : à convertir en format date.                                                    |
                | **INSEE_COM**                     | Code postal : plus de 300 modalités, à analyser pour regroupement.                                |
                | **IRIS**                          | Code IRIS : plus de 60 modalités, à analyser pour regroupement.                                   |
                | **CODE_IRIS**                     | Concaténation INSEE + IRIS : environ 500 modalités, à analyser pour regroupement.               |
                | **TYP_IRIS_x**                    | Modalités : `'H'`, `'Z'`, `'D'`. (H habitat, D divers et Z pour les communes non découpées en IRIS)                                                  |
                | **TYP_IRIS_y**                    | Modalités : `'H'`, `'Z'`. (H habitat, Z pour les communes non découpées en IRIS)                                                        |
                | **GRD_QUART**                     | Concaténation INSEE + numéro de quartier. À transformer en variable catégorielle.                 |
                | **UU2010**                        | Code postal version 2010 : plus de 44 modalités, à analyser pour regroupement.                   |
                | **REG**                          | Région À traiter comme catégorielle.                        |
                | **DEP**                          | Département À traiter comme catégorielle.                   |
                | **loyer_m2_median_n6**             | Moyenne du loyer au m² (base 6).                                                                     |
                | **loyer_m2_median_n7**             | Moyenne du loyer au m² (base 7).                                                                     |
                | **nb_log_n6**                     | Nombre de logements pour la moyenne `n6`.                                                             |
                | **nb_log_n7**                     | Nombre de logements pour la moyenne `n7`.                                                             |
                | **taux_rendement_n6**               | Taux de rendement basé sur la moyenne `n6`.                                                            |
                | **taux_rendement_n7**               | Taux de rendement basé sur la moyenne `n7`.                                                            |
                | **prix_m2_vente**                  | Prix au m² : prix_bien / surface. Variable lié à la cible donc à supprimer                                         |
                """)

        st.markdown("---")
        st.header("DataViz'")

        st.subheader("Variable cible : prix_bien")
        fig, ax = plt.subplots(figsize=(20, 8))
        sns.histplot(data=raw, x="prix_bien", bins=60, kde=True, ax=ax)
        ax.set_title("Distribution de la variable prix_bien")
        st.pyplot(fig)

        with st.expander("Corrélation des variables quantitatives"):
            numerical_vars = raw.select_dtypes(include=['int64', 'float64'])
            corr_matrix = numerical_vars.corr().abs()
            plt.figure(figsize=(20, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, annot_kws={"fontsize":7})
            plt.title("Matrice des coeff de corrélation en valeur absolue pour les variables numériques")
            st.pyplot(plt)

            st.subheader("Coefficient de corrélation avec la variable cible prix_bien")
            st.dataframe(corr_matrix['prix_bien'].drop('prix_bien').sort_values(ascending=False))

        st.subheader("Focus sur les variables surface et nb_pieces")

        c1, c2 = st.columns(2)

        fig, ax = plt.subplots(figsize=(20, 8))
        sns.regplot(data=raw, x='surface', y='prix_bien', order = 1, marker="x", color=".3", line_kws=dict(color="r"), x_estimator=np.median)
        ax.set_title('Prix médian par valeur de surface')
        c1.pyplot(fig)

        fig, ax = plt.subplots(figsize=(20, 8))
        sns.histplot(data=raw, x="surface", bins=60, kde=True, ax=ax)
        ax.set_title("Distribution de la variable surface")
        c2.pyplot(fig)


        fig, ax = plt.subplots(figsize=(20, 8))
        sns.regplot(data=raw[raw.surface <= 250], x='surface', y='prix_bien', order = 1, marker="x", color=".3", line_kws=dict(color="r"), x_estimator=np.median)
        ax.set_title('Prix médian par valeur de surface (Focus - 250m²)')
        c1.pyplot(fig)

        fig, ax = plt.subplots(figsize=(20, 8))
        sns.histplot(data=raw[raw.surface <= 250], x="surface", bins=60, kde=True, ax=ax)
        ax.set_title("Distribution de la variable surface (Focus - 250 m²)")
        c2.pyplot(fig)

        fig, ax = plt.subplots(figsize=(20, 8))
        sns.regplot(data=raw, x='nb_pieces', y='prix_bien', order = 1, marker="x", color=".3", line_kws=dict(color="r"), x_estimator=np.median)
        ax.set_title('Prix médian par valeur de nb_pieces')
        c1.pyplot(fig)

        fig, ax = plt.subplots(figsize=(20, 8))
        sns.histplot(data=raw, x="nb_pieces", bins=60, kde=True, ax=ax)
        ax.set_title("Distribution de la variable nb_pieces")
        c2.pyplot(fig)


        with st.expander("Corrélation des variables qualitatives"):
            st.subheader("Matrice de corrélation de Spearman sur les variables qualitatives")
            categorical_vars = raw.select_dtypes(include=['object']).columns
            categorical_as_num = raw[categorical_vars].apply(lambda col: col.astype('category').cat.codes)
            corr_spearman = pd.concat([categorical_as_num, raw['prix_bien']], axis=1).corr(method='spearman')
            corr_with_target = corr_spearman['prix_bien'].drop('prix_bien')
            corr_with_target_sorted = corr_with_target.abs().sort_values(ascending=False)
            st.dataframe(corr_with_target_sorted)

        st.subheader("Focus sur les variables typedebien_lite, ascenseur et TYP_IRIS_x")
        fig, ax = plt.subplots(figsize=(20, 8))
        sns.histplot(data=raw, x="prix_bien", hue="typedebien_lite", bins=60, kde=True, ax=ax)
        ax.set_title("Distribution de la variable prix_bien par type de bien Appartements / Maisons")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(20, 8))
        sns.histplot(data=raw, x="prix_bien", hue="ascenseur", bins=60, kde=True, ax=ax)
        ax.set_title("Distribution de la variable prix_bien par ascenseur False / True (Quelques NA)")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(20, 8))
        sns.histplot(data=raw, x="prix_bien", hue="TYP_IRIS_x", bins=60, kde=True, ax=ax)
        ax.set_title("Distribution de la variable prix_bien par TYP_IRIS_x")
        st.pyplot(fig)

        with st.expander("Positionnement des annonces sur la carte du Haut-Rhin"):
            if all(c in raw.columns for c in ["mapCoordonneesLatitude", "mapCoordonneesLongitude"]):
                map_df = raw.copy().reset_index().rename(columns={"index": "idannonce"})
                view = pdk.ViewState(latitude=float(map_df["mapCoordonneesLatitude"].median()), longitude=float(map_df["mapCoordonneesLongitude"].median()), zoom=8)
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position="[mapCoordonneesLongitude, mapCoordonneesLatitude]",
                    get_radius=80,
                    pickable=True,
                    get_fill_color=[0, 102, 255, 220]
                )
                st.pydeck_chart(pdk.Deck(
                    map_style=None,
                    initial_view_state=view,
                    layers=[layer],
                ))
            else:
                st.info("Carte indisponible : mapCoordonneesLatitude/mapCoordonneesLongitude manquantes.")



# ==========================================================
# SECTION 2
# ==========================================================
elif section == "2. Préprocessing & Feature engineering":
    st.header("2) Préprocessing & Feature engineering")

    with st.expander("2.1 Suppression de variables presque vide, conversion float vers int et variables bimodale en int"):
        st.code(
            """
    # suppression colonnes quasi vides / fuite cible
    col_na_prct = df_ventes_raw.isna().sum()/df_ventes_raw.shape[0] * 100
    to_delete = col_na_prct[col_na_prct > 80].index.to_list()
    to_delete.append('prix_m2_vente')
    to_delete.append('typedebien')
    to_delete.append('type_annonceur')
    to_delete += [ column for column in df_ventes_raw.columns if "n6" in column]

    def clean_df(df: pd.DataFrame):
        # On supprime les colonnes identifiées précédement
        df_cleaned = df.drop(to_delete, axis=1)

        # On transform en Int des colonnes déclarées en float mais n'ayant que des int
        valeurs_manq_resid_quanti = [col for  col in df_cleaned.select_dtypes(exclude='object').columns if df_cleaned[col].isna().sum() > 0]    
        col_float = df_cleaned[valeurs_manq_resid_quanti].select_dtypes(include='float64').columns
        for col in col_float:
            array_col = np.array(df_cleaned[df_cleaned[col].notna()][col]) 
            array_col_round = np.round(array_col)
            array_real_float = array_col[array_col != array_col_round]
            if len(array_real_float) == 0:
                df_cleaned[col] = df_cleaned[col].astype('Int64')
        # on transforme les colonnes quali n'ayant en fait que deux modalités
        if 'cave' in df_cleaned.columns:
            df_cleaned["cave"] = df_cleaned["cave"].astype('Int64')
        if 'ascenseur' in df_cleaned.columns:
            df_cleaned["ascenseur"] = df_cleaned["ascenseur"].astype('Int64')
        if 'logement_neuf' in df_cleaned.columns:
            df_cleaned["logement_neuf"] = df_cleaned["logement_neuf"].replace({'n': False, 'o': True}).astype('Int64')

        return df_cleaned

    df_ventes_clean = clean_df(df_ventes_raw)
    # 13 variables supprimées
    """, language="python")
        
    with st.expander("2.2 Train / Test split"):
        st.code(
            """
target_feature = 'prix_bien'
target = df_ventes_clean[target_feature]
data = df_ventes_clean.drop(target_feature, axis=1)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42) 
            """, language="python")

    with st.expander("2.3 KNNImputer custom pour remplir les NA quali et quanti"):
        st.code(
            """
# KNN (k=10) sur le type de bien, latitude, longitude et nombre de pièces
class KNNImputerCustom(BaseEstimator, TransformerMixin):
def __init__(self, type_col="typedebien_lite", lat_col="mapCoordonneesLatitude", lon_col="mapCoordonneesLongitude", pieces_col="nb_pieces", k=10):
    self.type_col = type_col
    self.lat_col = lat_col
    self.lon_col = lon_col
    self.pieces_col = pieces_col
    self.k = k
    self.fitted = False

def fit(self, X, y=None):
    self.df_train_ = X.copy()
    self.numeric_cols_ = X.select_dtypes(exclude="object").columns.tolist()
    self.default_mean_ = {col: X[col].mean() for col in self.numeric_cols_}
    self.imputer_ = KNNImputer()
    self.quali_cols_ = X.select_dtypes(include="object").columns.tolist()
    self.default_mode_ = {col: X[col].mode()[0] for col in self.quali_cols_}
    self.fitted = True
    
    return self

def transform(self, X):
    if not(self.fitted):
        raise NotFittedError("This KNNImputerCustomed instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
    X = X.copy()
    # First we fill the numeric columns
    for col in self.numeric_cols_:
        if X[col].isna().sum() == 0:
            continue
        cols = [col, self.pieces_col, self.lat_col, self.lon_col]
        df_extract = X[[self.type_col] + cols]
        df_extract_train = self.df_train_ [[self.type_col] + cols]
        for typ in ["a", "m"]:
            subset = df_extract[df_extract[self.type_col] == typ][cols]
            if subset.shape[0] == 0:
                continue
            apply_default = True
            subset_train = df_extract_train[df_extract_train[self.type_col] == typ][cols]
            if subset_train.shape[0] != 0:
                self.imputer_.fit(subset_train)
                imputed = self.imputer_.transform(subset)
                if imputed.shape[1] == len(cols):
                    imputed_df = pd.DataFrame(imputed, columns=cols, index=subset.index)
                    apply_default = False
            if apply_default:
                if X[col].dtype in ["int64", "Int64"]:
                    X.loc[subset.index, col] = self.default_mean_[col].astype("int64")
                else:
                    X.loc[subset.index, col] = self.default_mean_[col]
            else:
                if X[col].dtype in ["int64", "Int64"]:
                    X.loc[imputed_df.index, col] = imputed_df[col].round().astype("int64")
                else:
                    X.loc[imputed_df.index, col] = imputed_df[col]

    #Secondly we fill the qualitative columns
    index_na = X[X[self.quali_cols_].isna().any(axis=1)].index
    for idx in index_na:
        lat = X.loc[idx, self.lat_col]
        lon = X.loc[idx, self.lon_col]
        nbp = X.loc[idx, self.pieces_col]
        typ = X.loc[idx, self.type_col]
        neighbours = self.df_train_[
            (self.df_train_[self.pieces_col] == nbp) &
            (self.df_train_[self.type_col] == typ)
        ]
        distances = (lat - neighbours[self.lat_col])**2 + (lon - neighbours[self.lon_col])**2
        for col in self.quali_cols_:
            if pd.isna(X.loc[idx, col]):
                valid = neighbours[neighbours[col].notna()]
                if valid.shape[0] > 0:
                    idx_neigh = distances[valid.index].sort_values().iloc[:self.k].index
                    X.loc[idx, col] = neighbours.loc[idx_neigh][col].mode()[0]
                else:
                    X.loc[idx, col] = self.default_mode_[col]
    
    return X""",
            language="python"
        )

    with st.expander("2.4 Fit & Transform du KNNImputer"):
        st.code(
            """
knn_imputer_custom = KNNImputerCustom()
X_train_transformed = knn_imputer_custom.fit_transform(X_train)
X_test_transformed = knn_imputer_custom.transform(X_test)
            """, language="python"
        )

    with st.expander("2.5 Suppression des outliers (7% en dehors de [P01, P99])"):
        st.code(
            """
# X_train
X_train_outlier = X_train_transformed

# On ne fait la suppressions des outiliers que sur les colonnes pertinantes
numeric_cols = [col for col in X_train_outlier.select_dtypes(include=['number']).columns.tolist() if col not in ['mapCoordonneesLatitude', 'mapCoordonneesLongitude', 'dpeC', 'annee_construction', 'mensualiteFinance', 'nb_logements_copro', 'charges_copro', 'duree_int', 'INSEE_COM', 'IRIS', 'CODE_IRIS', 'GRD_QUART', 'loyer_m2_median_n7', 'nb_log_n7', 'taux_rendement_n7']]

index_to_drop = []

for col in numeric_cols:
    lower = max(X_train_outlier[col].quantile(0.01), 0)
    upper = max(X_train_outlier[col].quantile(0.99), 0)
    temp_index_to_drop = X_train_outlier[(X_train_outlier[col] < lower) | (X_train_outlier[col] > upper)].index.to_list()
    index_to_drop.extend(temp_index_to_drop)

X_train_outlier = X_train_outlier.drop(index_to_drop)
y_train_outlier = y_train.drop(index_to_drop)
""", language="python")
        
        st.code(
            """
# X_test
# X_test_outlier = X_test_transformed

# On ne fait la suppressions des outiliers que sur les colonnes pertinantes
numeric_cols = [col for col in X_test_outlier.select_dtypes(include=['number']).columns.tolist() if col not in ['mapCoordonneesLatitude', 'mapCoordonneesLongitude', 'dpeC', 'annee_construction', 'mensualiteFinance', 'nb_logements_copro', 'charges_copro', 'duree_int', 'INSEE_COM', 'IRIS', 'CODE_IRIS', 'GRD_QUART', 'loyer_m2_median_n7', 'nb_log_n7', 'taux_rendement_n7']]

index_to_drop = []

for col in numeric_cols:
    lower = max(X_test_outlier[col].quantile(0.01), 0)
    upper = max(X_test_outlier[col].quantile(0.99), 0)
    temp_index_to_drop = X_test_outlier[(X_test_outlier[col] < lower) | (X_test_outlier[col] > upper)].index.to_list()
    index_to_drop.extend(temp_index_to_drop)

X_test_outlier = X_test_outlier.drop(index_to_drop)
y_test_outlier = y_test.drop(index_to_drop)""",
            language="python"
        )

    with st.expander("2.6 Feature engineering"):

        with st.expander("Ajout du prix médian par commune / type de bien"):
            st.code(
                """
median_dict_commune_type = (
    X_train.assign(prix_bien=y_train).groupby(['INSEE_COM', 'typedebien_lite'])['prix_bien']
    .median()
    .to_dict()
)

median_global = y_train.median()

def get_median_commune_type(row):
    key = (row['INSEE_COM'], row['typedebien_lite'])
    return median_dict_commune_type.get(key, median_global)

X_train['prix_median_commune'] = (
    X_train[['INSEE_COM', 'typedebien_lite']]
    .apply(get_median_commune_type, axis=1)
)

X_test['prix_median_commune'] = (
    X_test[['INSEE_COM', 'typedebien_lite']]
    .apply(get_median_commune_type, axis=1)
)
""",
                language="python"
            )
        
        with st.expander("Ajout du prix médian par type de bien"):
            st.code(
                """
median_dict_type = (
    X_train.assign(prix_bien=y_train).groupby(['typedebien_lite'])['prix_bien']
    .median()
    .to_dict()
)

median_global = y_train.median()

def get_median_type(row):
    key = (row['typedebien_lite'])
    return median_dict_type.get(key, median_global)

X_train['prix_median_type'] = (
    X_train[['typedebien_lite']]
    .apply(get_median_type, axis=1)
)

X_test['prix_median_type'] = (
    X_test[['typedebien_lite']]
    .apply(get_median_type, axis=1)
)
""",
                language="python"
            )
        
        with st.expander("Ajout du prix médian par nombre de pièces / type de bien"):
            st.code(
                """
median_dict_pieces = (
    X_train.assign(prix_bien=y_train).groupby(['nb_pieces', 'typedebien_lite'])['prix_bien']
    .median()
    .to_dict()
)

median_global = y_train.median()

def get_median_dict_pieces(row):
    key = (row['nb_pieces'], row['typedebien_lite'])
    return median_dict_pieces.get(key, median_global)

X_train['prix_median_pieces'] = (
    X_train[['nb_pieces', 'typedebien_lite']]
    .apply(get_median_dict_pieces, axis=1)
)

X_test['prix_median_pieces'] = (
    X_test[['nb_pieces', 'typedebien_lite']]
    .apply(get_median_dict_pieces, axis=1)
)
""",
                language="python"
            )

        with st.expander("Ajout du prix médian par nombre d'étages / type de bien"):
            st.code(
                """
median_dict_nb_etages = (
    X_train.assign(prix_bien=y_train).groupby(['nb_etages', 'typedebien_lite'])['prix_bien']
    .median()
    .to_dict()
)

median_global = y_train.median()

def get_median_dict_nb_etages(row):
    key = (row['nb_etages'], row['typedebien_lite'])
    return median_dict_nb_etages.get(key, median_global)

X_train['prix_median_nb_etage'] = (
    X_train[['nb_etages', 'typedebien_lite']]
    .apply(get_median_dict_nb_etages, axis=1)
)

X_test['prix_median_nb_etage'] = (
    X_test[['nb_etages', 'typedebien_lite']]
    .apply(get_median_dict_nb_etages, axis=1)
)
""",
                language="python"
            )

        with st.expander("Ajout de l'année en OneHotEncoder"):
            st.code(
                """
X_train['date'] = pd.to_datetime(X_train['date'])
X_test['date'] = pd.to_datetime(X_test['date'])

X_train['annee'] = X_train['date'].dt.year
X_test['annee'] = X_test['date'].dt.year

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
annee_train = ohe.fit_transform(X_train[['annee']])
annee_test  = ohe.transform(X_test[['annee']])

annee_cols = ohe.get_feature_names_out(['annee'])

X_train = pd.concat([X_train, pd.DataFrame(annee_train, columns=annee_cols, index=X_train.index)], axis=1).drop(['date', 'annee'], axis=1)
X_test  = pd.concat([X_test,  pd.DataFrame(annee_test,  columns=annee_cols, index=X_test.index)], axis=1).drop(['date', 'annee'], axis=1)
""", language="python"
            )

        with st.expander("Parsing et OHE du champ exposition"):
            st.code(
                """
def parse_exposition(df):
    df = df.copy()
    df['exposition_clean'] = (
        df['exposition'].astype(str)
        .str.lower()
        .str.replace(r'[,/\\\-]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    df['expo_nord']  = df['exposition_clean'].str.contains(r'\bnord\b',  na=False).astype(int)
    df['expo_sud']   = df['exposition_clean'].str.contains(r'\bsud\b',   na=False).astype(int)
    df['expo_est']   = df['exposition_clean'].str.contains(r'\best\b',   na=False).astype(int)
    df['expo_ouest'] = df['exposition_clean'].str.contains(r'\bouest\b', na=False).astype(int)
    df['expo_inconnue'] = df['exposition_clean'].str.contains(r'0|nan', na=False).astype(int)
    return df

X_train = parse_exposition(X_train).drop(['exposition', 'exposition_clean'], axis=1)
X_test  = parse_exposition(X_test).drop(['exposition', 'exposition_clean'], axis=1)
""", language="python"
            )

        with st.expander("Parsing et OHE du champ chauffage_systeme"):
            st.code(
                """
def parse_chauffage_systeme(df):
    df = df.copy()
    df['chauffage_systeme_clean'] = (
        df['chauffage_systeme'].astype(str)
        .str.lower()
        .str.replace(r'[,/\\\-]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    df['chauf_radiateur']  = df['chauffage_systeme_clean'].str.contains(r'\bradiateur\b', na=False).astype(int)
    df['chauf_sol']        = df['chauffage_systeme_clean'].str.contains(r'\bsol\b', na=False).astype(int)
    df['chauf_convecteur'] = df['chauffage_systeme_clean'].str.contains(r'\bconvecteur\b', na=False).astype(int)
    df['chauf_poele_bois'] = df['chauffage_systeme_clean'].str.contains(r'poêle|poele', na=False).astype(int)
    df['chauf_pac']        = df['chauffage_systeme_clean'].str.contains(r'pompe à chaleur|pac', na=False).astype(int)
    df['chauf_clim_rev']   = df['chauffage_systeme_clean'].str.contains(r'climatisation', na=False).astype(int)
    df['chauf_cheminee']   = df['chauffage_systeme_clean'].str.contains(r'cheminée|cheminee', na=False).astype(int)
    df['chauf_inconnu']    = df['chauffage_systeme_clean'].str.contains(r'nan', na=False).astype(int)
    return df

X_train = parse_chauffage_systeme(X_train).drop(['chauffage_systeme', 'chauffage_systeme_clean'], axis=1)
X_test  = parse_chauffage_systeme(X_test).drop(['chauffage_systeme', 'chauffage_systeme_clean'], axis=1)
""", language="python"
            )

        with st.expander("Parsing et OHE du champ chauffage_energie"):
            st.code(
                """
def parse_chauffage_energie(df):
    df = df.copy()
    df['chauffage_energie_clean'] = (
        df['chauffage_energie'].astype(str)
        .str.lower()
        .str.replace(r'[,/\\\-]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    df['energie_gaz']   = df['chauffage_energie_clean'].str.contains(r'\bgaz\b', na=False).astype(int)
    df['energie_elec']  = df['chauffage_energie_clean'].str.contains(r'électrique|electrique', na=False).astype(int)
    df['energie_fioul'] = df['chauffage_energie_clean'].str.contains(r'\bfioul\b', na=False).astype(int)
    df['energie_bois']  = df['chauffage_energie_clean'].str.contains(r'\bbois\b', na=False).astype(int)
    df['energie_inconnue'] = df['chauffage_energie_clean'].str.contains(r'nan', na=False).astype(int)
    return df

X_train = parse_chauffage_energie(X_train).drop(['chauffage_energie', 'chauffage_energie_clean'], axis=1)
X_test  = parse_chauffage_energie(X_test).drop(['chauffage_energie', 'chauffage_energie_clean'], axis=1)
""", language="python"
            )

        with st.expander("Parsing et OHE du champ chauffage_mode"):
            st.code(
                """
def parse_chauffage_mode(df):
    df = df.copy()
    df['chauffage_mode_clean'] = (
        df['chauffage_mode'].astype(str)
        .str.lower()
        .str.replace(r'[,/\\\-]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    df['chauffage_mode_individuel'] = df['chauffage_mode_clean'].str.contains(r'\bindividuel\b', na=False).astype(int)
    df['chauffage_mode_collectif']  = df['chauffage_mode_clean'].str.contains(r'\bcollectif\b', na=False).astype(int)
    df['chauffage_mode_central']    = df['chauffage_mode_clean'].str.contains(r'\bcentral\b', na=False).astype(int)
    df['chauffage_mode_inconnu']    = df['chauffage_mode_clean'].str.contains(r'nan', na=False).astype(int)
    return df

X_train = parse_chauffage_mode(X_train).drop(['chauffage_mode', 'chauffage_mode_clean'], axis=1)
X_test  = parse_chauffage_mode(X_test).drop(['chauffage_mode', 'chauffage_mode_clean'], axis=1)
""", language="python"
            )

        with st.expander("Parsing et OHE du champ dpeL"):
            st.code(
                """
def parse_dpe(df):
    df = df.copy()
    df['dpe_A'] = df['dpeL'].str.contains(r'A', na=False).astype(int)
    df['dpe_B'] = df['dpeL'].str.contains(r'B', na=False).astype(int)
    df['dpe_C'] = df['dpeL'].str.contains(r'C', na=False).astype(int)
    df['dpe_D'] = df['dpeL'].str.contains(r'D', na=False).astype(int)
    df['dpe_E'] = df['dpeL'].str.contains(r'E', na=False).astype(int)
    df['dpe_F'] = df['dpeL'].str.contains(r'F', na=False).astype(int)
    df['dpe_G'] = df['dpeL'].str.contains(r'G', na=False).astype(int)
    df['dpe_inconnu'] = (~df['dpeL'].isin(list("ABCDEFG"))).astype(int)
    return df

X_train = parse_dpe(X_train).drop('dpeL', axis=1)
X_test  = parse_dpe(X_test).drop('dpeL', axis=1)
""", language="python"
            )

        with st.expander("Parsing et OHE du champ ges_class"):
            st.code(
                """
def parse_ges(df):
    df = df.copy()
    df['ges_A'] = df['ges_class'].str.contains(r'A', na=False).astype(int)
    df['ges_B'] = df['ges_class'].str.contains(r'B', na=False).astype(int)
    df['ges_C'] = df['ges_class'].str.contains(r'C', na=False).astype(int)
    df['ges_D'] = df['ges_class'].str.contains(r'D', na=False).astype(int)
    df['ges_E'] = df['ges_class'].str.contains(r'E', na=False).astype(int)
    df['ges_F'] = df['ges_class'].str.contains(r'F', na=False).astype(int)
    df['ges_G'] = df['ges_class'].str.contains(r'G', na=False).astype(int)
    df['ges_inconnu'] = (~df['ges_class'].isin(list("ABCDEFG"))).astype(int)
    return df

X_train = parse_ges(X_train).drop('ges_class', axis=1)
X_test  = parse_ges(X_test).drop('ges_class', axis=1)
""", language="python"
            )

        with st.expander("OHE sur le reste des variables catégorielles"):
            st.code(
                """
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
to_encode = X_train.select_dtypes(include=['object', 'category']).columns.to_list()

ohe_train = ohe.fit_transform(X_train[to_encode])
ohe_test  = ohe.transform(X_test[to_encode])

ohe_cols = ohe.get_feature_names_out(to_encode)
X_train_ohe = pd.DataFrame(ohe_train, columns=ohe_cols, index=X_train.index)
X_test_ohe  = pd.DataFrame(ohe_test,  columns=ohe_cols, index=X_test.index)

X_train = pd.concat([X_train.drop(columns=to_encode), X_train_ohe], axis=1)
X_test  = pd.concat([X_test.drop(columns=to_encode),  X_test_ohe], axis=1)
""", language="python")
            
        with st.expander("Standardisation des données"):
            st.code(
                """
df_scaler = StandardScaler()

train_columns = X_train.columns
train_index = X_train.index
test_index = X_test.index

X_train_scaled = pd.DataFrame(data=df_scaler.fit_transform(X_train), index=train_index, columns=train_columns)
X_test_scaled = pd.DataFrame(data=df_scaler.transform(X_test), index=test_index, columns=train_columns)
""", language="python"
            )
            
        st.success("""
Nous avons maintenant **97** features dans nos jeux de données X_train et X_test
""")




# ==========================================================
# SECTION 3
# ==========================================================
elif section == "3. Méthodes de sélection de features":
    st.header("3) Méthodes de sélection de features (PCA, Coef de LR, Backward Elimination, SelectKBest)")

    with st.expander("PCA - Analyse des composantes principales"):
        st.code(
            """
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
""", language="python"
        )

        pca = PCA()
        X_train_pca = pca.fit_transform(X_train_scaled)

        plt.figure(figsize=(20, 8))
        plt.xlim(0,60)
        plt.xlabel('Nombre de composantes')
        plt.ylabel('Part de variance expliquée')
        plt.axhline(y = 0.95, color ='r', linestyle = '--')
        plt.plot(pca.explained_variance_ratio_.cumsum())
        st.pyplot(plt)

        st.success(
            """
Avec la méthode **PCA**, moins de 60 features permettent d'expliquer 95% de la variance.
"""
        )

        with st.expander("Tableau de la contribution moyenne absolue des features"):
            feature_contribution = pca.components_.T* np.sqrt(pca.explained_variance_)
            df_feature_contribution = pd.DataFrame(data=feature_contribution, index = X_train.columns).abs()
            df_feature_contribution['contribution_moyenne_absolue'] = df_feature_contribution.mean(axis=1)

            st.dataframe(df_feature_contribution['contribution_moyenne_absolue'].sort_values(ascending=False))

        with st.expander("Focus sur les deux premières composantes (20% de la variance expliquée)"):
            st.success(
                """
D'après nos analyses de corrélation avec la variable cible, nous identifions graphiquement **deux axes rouges** suivant le prix des biens.  
Nos variables sont ainsi plus ou moins explicatives selon la longeur et la direction des segments.
"""
            )
            
            pca_prince = prince.PCA(n_components=2)
            pca_prince = pca_prince.fit(X_train_scaled)
            corr = pca_prince.column_correlations


            fig, axes = plt.subplots(figsize=(10, 10))
            axes.set_xlim(-1, 1)
            axes.set_ylim(-1, 1)

            # Affichage des variables
            for j in X_train_scaled.columns:
                if max(np.abs(corr.loc[j, 0]), np.abs(corr.loc[j, 1])) > 0:
                    plt.annotate(j, (corr.loc[j, 0], corr.loc[j, 1]), color='#091158')
                    plt.arrow(0, 0, corr.loc[j, 0]*0.95, corr.loc[j, 1]*0.95, alpha=0.5, head_width=0.03, color='b')

            # Ajout des axes
            plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
            plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)

            plt.text(-1, -0.35, "← ← Axe de prix croissant ← ←", ha='left', va='bottom', transform_rotates_text=True, rotation=20, rotation_mode='anchor', color='red')
            plt.plot([-1,1], [-0.35,0.35], color='red', linestyle='--', linewidth=1)
            plt.text(-0.65, 1, "← ← Axe de prix croissant ← ←", ha='left', va='bottom', transform_rotates_text=True, rotation=-55, rotation_mode='anchor', color='red')
            plt.plot([-1,1], [1.5,-1.5], color='red', linestyle='--', linewidth=1)

            # Cercle et légendes
            cercle = plt.Circle((0, 0), 1, color='#16E4CA', fill=False)
            axes.add_artist(cercle)
            plt.xlabel('Component 0')
            plt.ylabel('Component 1')
            st.pyplot(fig)

    with st.expander("Coefficients d'une Regression Linéaire"):
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)

        columns_coef = dict(zip(X_train_scaled.columns, np.abs(lr.coef_.tolist())))
        columns_coef_trie = dict(sorted(columns_coef.items(), key=lambda item: item[1], reverse=True))

        df_resultats = pd.DataFrame([
            {'Feature': feature, 'Coefficient en valeur absolue': coef}
            for i, (feature, coef) in enumerate(columns_coef_trie.items(), start=1)
        ])

        st.code(
            """
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

columns_coef = dict(zip(X_train_scaled.columns, np.abs(lr.coef_.tolist())))
columns_coef_trie = dict(sorted(columns_coef.items(), key=lambda item: item[1], reverse=True))

df_resultats = pd.DataFrame([
    {'Feature': feature, 'Coefficient en valeur absolue': coef}
    for i, (feature, coef) in enumerate(columns_coef_trie.items(), start=1)
])
""", language="python"
        )

        st.dataframe(df_resultats)

    with st.expander("Backward Elimination"):
        selected_features = ['surface', 'surface_terrain', 'nb_pieces', 'mensualiteFinance', 'eau',
       'bain', 'mapCoordonneesLatitude', 'mapCoordonneesLongitude',
       'nb_etages', 'annee_construction', 'nb_toilettes', 'ascenseur',
       'logement_neuf', 'INSEE_COM', 'CODE_IRIS', 'UU2010',
       'prix_median_commune', 'prix_median_type', 'prix_median_pieces',
       'annee_2019', 'annee_2020', 'annee_2022', 'annee_2023', 'expo_sud',
       'expo_est', 'chauf_sol', 'chauf_pac', 'chauffage_mode_individuel',
       'dpe_A', 'dpe_B', 'dpe_C', 'dpe_D', 'dpe_E', 'dpe_inconnu',
       'typedetransaction_v', 'annonce_exclusive_Oui', 'categorie_annonceur_b',
       'categorie_annonceur_ca', 'categorie_annonceur_cm', 'TYP_IRIS_y_Z']

        st.code(
            """
lr = LinearRegression()
sfs = SequentialFeatureSelector(
    lr,
    n_features_to_select=40,
    direction="backward",
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="neg_mean_squared_error"
)

sfs.fit(X_train_scaled, y_train.values.ravel())

# Colonnes retenues
mask = sfs.get_support()
selected_features = X_train_scaled.columns[mask]
""", language="python"
        )

        st.dataframe(selected_features)

    with st.expander("SelectKBest"):
        kbest_features = ['etage', 'surface', 'surface_terrain', 'nb_pieces', 'balcon', 'bain',
       'dpeC', 'mapCoordonneesLongitude', 'nb_etages', 'places_parking',
       'nb_toilettes', 'ascenseur', 'nb_logements_copro', 'logement_neuf',
       'IRIS', 'UU2010', 'loyer_m2_median_n7', 'nb_log_n7',
       'taux_rendement_n7', 'prix_median_commune', 'prix_median_type',
       'prix_median_pieces', 'prix_median_nb_etage', 'energie_fioul',
       'chauffage_mode_individuel', 'chauffage_mode_collectif', 'dpe_A',
       'ges_A', 'typedetransaction_v', 'typedetransaction_vp',
       'annonce_exclusive_0', 'annonce_exclusive_Oui', 'categorie_annonceur_a',
       'categorie_annonceur_cm', 'typedebien_lite_a', 'typedebien_lite_m',
       'TYP_IRIS_x_H', 'TYP_IRIS_x_Z', 'TYP_IRIS_y_H', 'TYP_IRIS_y_Z']
        
        st.code(
            """
# Choix de K
k = 40

selector = SelectKBest(score_func=f_regression, k=k)
selector.fit(X_train_scaled, y_train.values.ravel())

# Colonnes retenues
mask = selector.get_support()
kbest_features = X_train_scaled.columns[mask]
""", language="python"
        )

        st.dataframe(kbest_features)

    st.success(
        """
Features communes entre les méthodes de sélections : **13**
* prix_median_type
* nb_pieces
* nb_etages
* dpe_A
* logement_neuf
* prix_median_pieces
* prix_median_commune
* nb_toilettes
* categorie_annonceur_cm
* bain
* annonce_exclusive_Oui
* surface
* UU2010
"""
    )

    st.info(
        """
Nous allons vérifier par la suite si les 13 features communes suffisent à l'entrainement de nos modèles et comparer les performances avec l'ensemble des features.
"""
    )



# ==========================================================
# SECTION 4
# ==========================================================
elif section == "4. Modélisation (LR, RFR & XGBR)":
    st.header("4) Modélisation : expérimentations avec plusieurs modèles")

    st.success(
        """
Afin de juger la performance des modèles, nous utiliserons les métrique suivante :
* **MAE (Mean Absolute Error)** : mesure la moyenne des erreurs absolues entre les valeurs prédites et les valeurs réelles. Elle indique à quel point, en moyenne, les prédictions sont éloignées des valeurs réelles.
* **RMSE (Root Mean Squared Error)** : calcule la racine carrée de la moyenne des carrés des erreurs. Elle donne une idée de l'ampleur des erreurs, en mettant davantage l'accent sur les grandes erreurs.
* **R2 (Coefficient of determination)** : indique la proportion de la variance des données expliquée par le modèle. Un R² proche de 1 signifie que le modèle explique bien les variations des données.
"""
    )

    with st.expander("LinearRegression()"):
        st.code(
            """
model_lr = LinearRegression()

model_lr.fit(X_train_scaled, y_train)

y_pred = model_lr.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
"""
        )

        st.subheader("Train avec l'ensemble des features")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", 58274.70)
        c2.metric("RMSE", 86314.96)
        c3.metric("R2", 0.6970)
        st.markdown("---")
        st.subheader("Train avec les 13 features communes")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", 64912.55)
        c2.metric("RMSE", 95186.15)
        c3.metric("R2", 0.6315)


    with st.expander("RandomForestRegressor()"):
        st.code(
            """
model_rfr = RandomForestRegressor()

model_rfr.fit(X_train_scaled, y_train)

y_pred = model_rfr.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
"""
        )

        st.subheader("Train avec l'ensemble des features")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", 26592.13)
        c2.metric("RMSE", 55222.71)
        c3.metric("R2", 0.8760)
        st.markdown("---")
        st.subheader("Train avec les 13 features communes")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", 54009.51)
        c2.metric("RMSE", 86470.79)
        c3.metric("R2", 0.6959)


    with st.expander("XGBRegressor()"):
        st.code(
            """
model_xgbr = XGBRegressor()

model_xgbr.fit(X_train_scaled, y_train)

y_pred = model_xgbr.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
"""
        )

        st.subheader("Train avec l'ensemble des features")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", 28624.56)
        c2.metric("RMSE", 53543.30)
        c3.metric("R2", 0.8834)
        st.markdown("---")
        st.subheader("Train avec les 13 features communes")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", 55902.75)
        c2.metric("RMSE", 86278.78)
        c3.metric("R2", 0.6972)


    with st.expander("RandomForestRegressor() avec tuning des parametres"):
        st.code(
            """
model_rfr = RandomForestRegressor(
    n_estimators=1000,
    min_samples_split=5, 
    min_samples_leaf=2, 
    max_features=0.5, 
    max_depth=None, 
    criterion='squared_error', 
    bootstrap=False)

model_rfr.fit(X_train_scaled, y_train)

y_pred = model_rfr.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
"""
        )

        st.subheader("Train avec l'ensemble des features")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", 25745.59)
        c2.metric("RMSE", 53158.89)
        c3.metric("R2", 0.8850)


    with st.expander("XGBRegressor() avec tuning des parametres"):
        st.code(
            """
model_xgbr_tunning = XGBRegressor(
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

model_xgbr_tunning.fit(X_train_scaled, y_train)

y_pred = model_xgbr_tunning.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
"""
        )

        st.subheader("Train avec l'ensemble des features")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", 25006.34)
        c2.metric("RMSE", 48638.29)
        c3.metric("R2", 0.9038)



    st.markdown("---")
    st.subheader("Synthèse des performances de nos modèles :")
    rows = []
    for name in models.keys():
        y_pred = pred_df[name].values
        m = compute_metrics(y_test.values, y_pred)
        rows.append([name, m["MAE"], m["RMSE"], m["R2"]])

    res = pd.DataFrame(rows, columns=["Modèle", "MAE", "RMSE", "R2"]).sort_values("MAE")
    st.dataframe(res.style.format({"MAE":"{:,.0f}", "RMSE":"{:,.0f}", "R2":"{:.4f}"}))


    st.info(
        """
Liste des expérimentations non fructueuses :
* Prédiction du log du prix
* Prédiction d'une classe de prix (Classification)
* OverSampling et UnderSampling sur les variables les plus discriminantes niveau performances

Dans la section suivante, nous nous concentrerons sur l'analyse des écarts entre les prédictions et la réalité.
"""
    )

    

# ==========================================================
# SECTION 5
# ==========================================================
elif section == "5. Comparaisons des performances (Focus : surface et type de bien)":
    st.header("5) Comparaisons des performances de nos modèles sur les critères les plus discrimiants")

    focus_type = st.selectbox("Focus type de bien", ["Tous", "🏢 Appartements", "🏠 Maisons"])
    surface_max = X_test.surface.max()
    surface_focus = st.slider("Focus surface", 0, surface_max, (0, surface_max))

    idx = X_test.index
    mask = pd.Series(True, index=idx)

    if focus_type == "🏢 Appartements" and "typedebien_lite_a" in X_test.columns:
        mask &= X_test["typedebien_lite_a"] == 1
    elif focus_type == "🏠 Maisons" and "typedebien_lite_m" in X_test.columns:
        mask &= X_test["typedebien_lite_m"] == 1

    if "surface" in X_test.columns:
        mask &= X_test["surface"].between(surface_focus[0], surface_focus[1], inclusive="both")

    selected_idx = mask[mask].index
    st.metric("Nombre d'annonces de tests correspondants aux filtres :", len(selected_idx))
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


    st.info(
        """
L'écart en valeur absolu moyen est bien plus bas sur les 🏢 Appartements que sur les 🏠 Maisons (ratio x 5). 

Le modèle le plus performant, **XGBRegressor_tuned**, convient parfaitement à la prédiction du prix des 🏢 Appartements dans le Haut-Rhin !


Au dela de 120 m² les performances de nos prédictions diminuent pour les 🏠 Maisons :
* Ratio x 2 entre les 🏠 Maisons <=120m² (MAE 26K€) et > 120m² (MAE 52K€)
* Ratio x 3 entre les 🏠 Maisons <=120m² (MAE 26K€) et > 170m² (MAE 78K€)
* Ratio x 4 entre les 🏠 Maisons <=120m² (MAE 26K€) et > 250m² (MAE 94K€)



Au dela de 120 m² les performances de nos prédictions diminuent **fortement** pour les 🏢 Appartements :
* Ratio x 6 entre les 🏢 Appartements <=120m² (MAE 6K€) et > 120m² (MAE 36K€)
* Ratio x 14 entre les 🏢 Appartements <=120m² (MAE 6K€) et > 170m² (MAE 75K€)
"""
    )

# ==========================================================
# SECTION 6
# ==========================================================
elif section == "6. Interprétabilité avec SHAP":
    st.header("6) Interprétabilité avec SHAP")

    model = models["XGBRegressor_tuned"]
    X_sample = X_test_scaled.sample(n=500)

    with st.spinner("Calcul SHAP..."):
        explainer = shap.Explainer(model, X_train_scaled.sample(n=2000))
        shap_values = explainer(X_sample)

    st.text(
        """
Sélection d'un sample aléatoire de X_train de 500 éléments pour déterminer l'importance de nos features."""
    )
    fig = plt.figure(figsize=(8, 4))
    shap.plots.bar(shap_values, max_display=15, show=False)
    st.pyplot(fig)

    st.text(
        """
Distinction par type de bien.
""")
    type = ["Appartements" if shap_values[i, "typedebien_lite_m"].data > 0 else "Maisons" for i in range(shap_values.shape[0])]
    fig = plt.figure(figsize=(8, 4))
    shap.plots.bar(shap_values.cohorts(type).abs.mean(0), max_display=20, show=False)
    st.pyplot(fig)

    st.text(
        """
Beeswarm d'un sample aléatoire de X_test de 500 éléments pour comprendre graphiquement l'impact de nos features sur la valeur prédite."""
    )
    fig = plt.figure(figsize=(8, 4))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    st.pyplot(fig)


# ==========================================================
# SECTION 7
# ==========================================================
elif section == "7. Limites & perspectives":
    st.header("7) Limites & perspectives")
    st.markdown("""
    **Limites**
    - Prix d’annonce vs prix de transaction
    - Données bruitées, hétérogènes
    - Sensibilité aux biens atypiques
    
    **Perspectives**
    - Ajout données externes pour mieux prédire les maisons
    """)

# ==========================================================
# SECTION 8
# ==========================================================
elif section == "8. Démonstration intéractive":
    st.header("8) Démonstration intéractive")

    # Choix modèle
    default_model = "XGBRegressor_tuned" if "XGBRegressor_tuned" in models else list(models.keys())[0]
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

    with st.expander("Focus sur un bien"):
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

        # SHAP waterfall unitaire
        st.subheader("SHAP Waterfall (annonce sélectionnée)")
        with st.spinner("Calcul SHAP unitaire..."):
            explainer = shap.Explainer(model, X_train_scaled.iloc[:1000])
            x_one = X_test_scaled.loc[[selected_idx]]
            sv = explainer(x_one)

        fig = plt.figure(figsize=(9, 5))
        shap.plots.waterfall(sv[0], max_display=15, show=False)
        st.pyplot(fig)


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

        view = pdk.ViewState(latitude=float(map_df["mapCoordonneesLatitude"].median()), longitude=float(map_df["mapCoordonneesLongitude"].median()), zoom=8)
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

st.markdown("---")
st.caption("Soutenance Projet Compagnon Immobilier — Data Scientist #2 Orange")
