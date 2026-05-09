# ML Anomaly Detection for IoT — P05

> Détection d'anomalies thermiques non supervisée sur capteurs DHT11 industriels.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

---

## Description

Ce projet met en œuvre un pipeline complet de **détection d'anomalies non supervisée** sur des données de capteurs IoT thermiques (DHT11). Il compare 5 algorithmes d'apprentissage non supervisé et applique une stratégie de **calibrage des seuils** pour atteindre l'objectif **Recall ≥ 80 %** sur un dataset Kaggle volontairement difficile.

### Contexte industriel
Capteurs déployés sur équipements industriels (armoires électriques, moteurs, transformateurs). L'objectif est de détecter automatiquement des comportements anormaux : surchauffes, dérives thermiques, spikes de tension, refroidissement anormal.

---

## Dataset

Source : [Kaggle — IoT T-Sensor Dataset for Anomaly Detection](https://www.kaggle.com/datasets/diaealaouisoulimani/iot-t-sensor-dataset-for-anomaly-detection)

| Élément | Détail |
|---|---|
| Fichier principal | `synthetic_iot_dataset_challenging.csv` |
| Fichier référence | `clean_unlabeled_autoencoder.csv` |
| Observations | 3 000 |
| Variables | 5 (`Device_ID`, `Temperature`, `Humidity`, `Battery_Level`, `Anomaly`) |
| Capteurs | 4 × DHT11 (A, B, C, D) |
| Anomalies | 522 (17.4 %) |
| État | Variables numériques **déjà standardisées** (z-score) |

> Les fichiers CSV ne sont pas inclus dans ce dépôt. Téléchargez-les depuis Kaggle et placez-les à la racine du projet avant d'exécuter le notebook.

---

## Modèles implémentés

### Modèles obligatoires
1. **Isolation Forest** — isolation par arbres aléatoires
2. **K-Means + Elbow** — clustering avec choix automatique de k via la courbe coude
3. **PCA** — réduction 2D + détection par erreur de reconstruction

### Modèles littérature (choix justifié)
4. **LOF** (Local Outlier Factor) — densité locale, anomalies contextuelles
5. **DBSCAN** — clustering par densité, outliers automatiques

### Stratégie d'ensemble
- **Vote majoritaire (≥ 3/5)** sur les modèles calibrés

---

## Approche méthodologique

```
┌─────────────────────────────────────────────────────────────┐
│  1. EDA (avant modélisation)                                │
│     → séries temporelles, distributions, corrélation        │
├─────────────────────────────────────────────────────────────┤
│  2. Feature engineering : 3 → 12 variables                  │
│     → différences, magnitude, interactions, one-hot device  │
├─────────────────────────────────────────────────────────────┤
│  3. Entraînement des 5 modèles non supervisés               │
├─────────────────────────────────────────────────────────────┤
│  4. Calibrage des seuils via courbes Precision-Recall       │
│     → F1 max sous contrainte recall ≥ 0.80                  │
├─────────────────────────────────────────────────────────────┤
│  5. Ensemble par vote majoritaire                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Résultats

### Modèles calibrés (F1-max sous contrainte `recall ≥ 0.80`)

| Modèle | Recall | Précision | F1-Score | AUC-PR |
|---|:---:|:---:|:---:|:---:|
| Isolation Forest | 0.960 | 0.177 | 0.298 | 0.162 |
| K-Means (k=5) | 0.996 | 0.175 | 0.298 | 0.170 |
| PCA (reconstruction) | 0.990 | 0.175 | 0.298 | 0.183 |
| **LOF (k=25)** | 0.828 | 0.198 | 0.319 | **0.214** |
| **DBSCAN** | 0.866 | **0.200** | **0.325** | 0.210 |

### Ensemble (vote majoritaire ≥ 3/5)

| Métrique | Valeur |
|---|:---:|
| Recall | **0.981** |
| Précision | 0.178 |
| F1-Score | 0.302 |

### Conclusion

- **Objectif `Recall ≥ 0.80` ATTEINT pour les 5 modèles**
- **DBSCAN** offre le meilleur compromis (F1 = 0.325)
- **LOF** est le meilleur modèle au sens AUC-PR (indépendant du seuil)
- L'**ensemble par vote** maximise la couverture (recall = 98 %)

---

## Notebook — Plan détaillé

| Section | Description |
|---|---|
| 0. Imports et configuration | Libraires, paramètres globaux |
| 1. Chargement des datasets | Lecture des CSV Kaggle |
| 2. Analyse Exploratoire (EDA) | Visualisation **avant** toute modélisation |
| 3. Prétraitement & feature engineering | 3 → 12 variables |
| 4. Modèle 1 — Isolation Forest | obligatoire |
| 5. Modèle 2 — K-Means + Elbow | obligatoire |
| 6. Modèle 3 — PCA 2D | obligatoire |
| 7. Modèle 4 — LOF | littérature |
| 8. Modèle 5 — DBSCAN | littérature |
| 9. Calibrage des seuils & comparaison | Courbes PR, AUC-PR, ensembles |
| 10. Visualisation temporelle finale | Graphique principal |
| 11. Synthèse et conclusion | Tableau récapitulatif |

---

## Installation

### Prérequis
- Python 3.10 ou supérieur
- pip

### Installation des dépendances

```bash
git clone https://github.com/hechmi06/ML_AnomalyDetection-for-IoT.git
cd ML_AnomalyDetection-for-IoT
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Téléchargement des datasets

1. Aller sur [Kaggle — IoT T-Sensor Dataset](https://www.kaggle.com/datasets/diaealaouisoulimani/iot-t-sensor-dataset-for-anomaly-detection)
2. Télécharger les deux fichiers CSV
3. Les placer à la racine du projet

### Lancer le notebook

```bash
jupyter notebook P05_Detection_Anomalies_Thermiques_IoT.ipynb
```

---

## Technologies utilisées

- **Python 3.10**
- **NumPy** / **Pandas** — manipulation de données
- **scikit-learn** — modèles ML (`IsolationForest`, `KMeans`, `PCA`, `LocalOutlierFactor`, `DBSCAN`)
- **Matplotlib** / **Seaborn** — visualisations
- **Jupyter Notebook** — environnement d'analyse

---

## Auteur

**hechmi06** — Projet académique P05  
GitHub : [@hechmi06](https://github.com/hechmi06)
