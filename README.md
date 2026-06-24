# 🏠 Projet Compagnon Immobilier


Projet de **prédiction de prix de biens immobiliers** dans le **Haut-Rhin (68)**, à partir d’annonces de ventes immobilières.

Dans ce projet Git on retrouvera nottament l'application **Streamlit** (Streamlit_Soutenance.py) et le **Notebook** regroupant toutes les étapes du projet (Notebook_global.ipynb).

## 🍊 Membres du projet

* Laila BENHASSA
* Hugues OUDEVILLE
* Matthieu ROBERT

---

## 📌 Objectif

L’objectif du projet est de **prédire le prix d’un bien immobilier** à partir de ses caractéristiques, par exemple :

- surface
- nombre de pièces
- localisation
- type de bien
- année
- données énergétiques
- exposition
- chauffage
- variables agrégées créées lors du feature engineering

---

## 🗂️ Données

Le projet s’appuie sur un jeu de données d’annonces immobilières contenant environ **27 000 annonces** de **maisons** et **appartements** dans le **Haut-Rhin**, entre **2019 et 2023**.

Source mentionnée dans le projet :

- Repository GitHub : [klopstock-dviz/immo_vis](https://github.com/klopstock-dviz/immo_vis)
- Fichier utilisé : `data/ech_annonces_ventes_68.csv`

---

## 🚀 Contenu du Streamlit

L’application Streamlit est organisée en 8 sections :

1. **Contexte & données**
2. **Préprocessing & Feature engineering**
3. **Méthodes de sélection de features**
4. **Modélisation (LR, RFR & XGBR)**
5. **Comparaisons des performances**
6. **Interprétabilité avec SHAP**
7. **Limites & perspectives**
8. **Démonstration interactive**

Elle inclut également :
- un **timer de section**
- un **timer global**
- des visualisations avec **Matplotlib**, **Seaborn** et **PyDeck**
- des comparaisons de performances par modèle
- une exploration interactive des prédictions

---

## 🐍 Contenu du Notebook

L'ensemble du code Python a été condensé dans un Notebook contenant les sections suivantes :

1. **Import des modules**
2. **Import des données brutes**
3. **Découverte des données & DataViz**
4. **Preprocessing des données**
5. **Feature Engineering**
6. **Feature Selection**
7. **Modélisation : Régression**

---

## 🧠 Modèles utilisés

Les modèles évalués dans le projet sont :

- **LinearRegression**
- **RandomForestRegressor**
- **XGBRegressor**
- **RandomForestRegressor tuned**
- **XGBRegressor tuned**

### Métriques utilisées
- **MAE**
- **RMSE**
- **R²**

---

## 🛠️ Stack technique

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **XGBoost**
- **SHAP**
- **PyDeck**
- **Prince**
