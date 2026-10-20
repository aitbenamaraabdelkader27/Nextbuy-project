> Historical EPITECH project documentation. For the restored application, verified metrics and installation instructions, see [the root README](../README.md). The original notebook and `src/modeling.py` are retained as academic evidence; use [the executed restoration notebook](../notebooks/analysis.ipynb) for the corrected workflow. The dashboard launcher below now opens the restored app.

# DASHBOARD_GUIDE 
python3 -m venv .venv
source .venv/bin/activate
## Installation
pip install -r requirements.txt
## Lancement 
streamlit run app.py

# NextBuy

## Présentation du projet

NextBuy est un projet d’analyse de données et de modélisation prédictive à partir de données de commandes de produits alimentaires.

L’objectif principal est de transformer des données transactionnelles brutes en résultats exploitables, à la fois pour l’analyse métier et pour la prédiction.  
Le projet s’articule autour de trois axes :

- **prétraitement des données** : chargement, fusion et nettoyage des différents fichiers CSV ;
- **analyse exploratoire (EDA)** : étude des comportements d’achat à l’aide de statistiques et de visualisations ;
- **machine learning** : construction et comparaison de modèles capables de prédire si un produit sera recommandé à nouveau (`reordered`).

En complément du notebook principal, nous avons également développé un **dashboard Streamlit** permettant d’explorer les principaux résultats de manière plus interactive.

---

## Contenu du projet

Le projet contient deux livrables principaux :

### 1. Le notebook Jupyter
Le notebook constitue le livrable principal du projet.  
Il contient :
- le chargement et la fusion des jeux de données ;
- l’analyse exploratoire des comportements clients ;
- les insights business formulés sous forme de questions ;
- la partie machine learning avec entraînement, test et comparaison de plusieurs modèles ;
- les conclusions finales.

### 2. Le dashboard Streamlit
Le dashboard permet de présenter le projet sous une forme plus visuelle et interactive.  
Il permet notamment :
- de visualiser les principales statistiques du dataset ;
- d’explorer plusieurs insights business ;
- d’afficher les analyses de co-purchase ;
- de lancer la partie machine learning depuis une interface simple.

---

# Contribution – Analyse des données (Insights)

## Présentation
Dans ce projet d’analyse de données, ma contribution principale a été de travailler sur la partie **exploration et visualisation des données** afin d’extraire plusieurs **insights pertinents** à partir des datasets fournis

L’objectif était de comprendre certains comportements présents dans les données et de les représenter de manière **claire, visuelle et compréhensible** grâce à des graphiques réalisés en Python



## Méthodologie
Pour réaliser ce travail, j’ai suivi plusieurs étapes :

### 1. Exploration des datasets
J’ai commencé par explorer les fichiers CSV afin de comprendre leur structure, les colonnes disponibles et les informations exploitables

Cela m’a permis d’identifier les variables intéressantes pour l’analyse comme par exemple :

- les commandes  
- les produits  
- la taille des paniers  

### 2. Nettoyage et validation des données
Avant de faire les analyses, j’ai ajouté des **vérifications de validation des datasets** afin de m’assurer que :

- les fichiers existent  
- les colonnes nécessaires sont présentes  
- les données peuvent être utilisées correctement  

### 3. Création des visualisations  
Ensuite, j’ai développé plusieurs **graphiques avec Python (Matplotlib et Pandas)** afin de représenter les informations importantes dans les données

Chaque graphique correspond à un **insight spécifique**

J’ai essayé de créer des visualisations qui soient :

- simples  
- lisibles  
- faciles à comprendre rapidement  


## Travail d’équipe
Le projet a été réalisé en groupe, chaque membre travaillant sur différentes parties

De mon côté, j’ai principalement contribué à :

- l’analyse des datasets  
- la validation des données  
- la création des insights et des graphiques  

## Conclusion
Ce travail m’a permis de pratiquer :

- l’analyse de données  
- la manipulation de datasets CSV  
- la création de visualisations avec Python  

Cela m’a également permis de mieux comprendre comment **extraire des informations utiles à partir de données brutes**

## Partie EDA Avancée + Insights Business

Cette partie du projet vise à explorer les données en profondeur afin de dégager des insights business exploitables, qui complètent la partie machine learning.

## Objectifs de l’EDA

L’objectif principal est de :

identifier les produits et départements avec les taux de réachat les plus élevés

analyser la relation entre taux de reorder et volume de commandes

repérer les produits fréquemment ajoutés en premier dans le panier (add_to_cart_order = 1)

détecter les paires de produits souvent achetées ensemble et suggérer des opportunités de bundles

## Contenu et méthodologie

L’EDA a été réalisée dans le notebook principal, section EDA 2, et peut être complétée par le fichier optionnel docs/insights.md.

Pour chaque analyse, nous avons généré :

des statistiques descriptives (taux, counts, proportions)

des visualisations graphiques (bar plots, heatmaps, scatter plots)

des interprétations business permettant de formuler des recommandations concrètes

## Analyses réalisées 

1. Analyse des produits ré-achetés (reordered)

.Identification des produits et départements avec les plus forts taux de reorder

.Comparaison du reorder vs volume, pour détecter les produits stratégiques à fort impact

2. Analyse “First in Cart”

.Sélection des produits achetés en premier dans le panier

.Détermination des top produits et interprétation des comportements clients

3. Co-purchase pairs

.Détection des produits fréquemment achetés ensemble

.Proposition d’idées de bundles ou promotions croisées

4. Questions business et réponses

.Formulation de 4 questions pertinentes basées sur les données

.Réponses argumentées avec statistiques et visualisations pour appuyer les recommandations

## Résultats attendus

.4 sections “Insight” avancées

.Chaque insight comprend :

.des statistiques claires et interprétables

.au moins une visualisation ou tableau illustrant le résultat

.Ces insights peuvent être exploités pour optimiser les promotions, le placement produit, ou les stratégies de fidélisation

## Remarque sur les performances

Certaines analyses traitent des datasets volumineux ; pour assurer l’exécution fluide :

.des structures allégées ont été utilisées

.des échantillons représentatifs ont été sélectionnés

.des traitements optimisés ont été appliqués

## Partie Machine Learning

La partie machine learning du projet a pour objectif de prédire si un produit a des chances d’être recommandé à nouveau par un client.

### Variable cible
La variable cible utilisée est :

- `reordered`

Les modèles cherchent donc à prédire si un produit présent dans une commande sera probablement racheté ou non.

### Préparation des données pour le ML
Pour la partie machine learning, un dataframe allégé a été construit afin de conserver uniquement les colonnes utiles au modèle.  
Cette étape permet de réduire la consommation mémoire et de stabiliser l’entraînement.

Les variables exploitées dans le pipeline sont :

- statistiques produit :
  - taux de reorder par produit
  - nombre de commandes par produit
- statistiques utilisateur :
  - taux de reorder par utilisateur
  - nombre de commandes par utilisateur
- contexte de commande :
  - `add_to_cart_order`
  - `order_hour_of_day`
  - `order_dow`
  - `days_since_prior_order`
  - `order_number`
- variables catégorielles :
  - `department`
  - `aisle`

### Stratégie de split
Pour éviter les fuites de données et simuler une situation réaliste, la **dernière commande de chaque utilisateur** est utilisée comme jeu de test.

### Modèles entraînés
Deux modèles de classification ont été comparés :

- **Logistic Regression**
- **Random Forest**

### Évaluation
Les performances ont été évaluées à l’aide de :

- **ROC-AUC**
- **Average Precision**
- **classification report**

Cette comparaison permet d’identifier le modèle le plus pertinent pour anticiper le comportement de réachat.

### Conclusion ML
Cette partie permet de montrer comment des signaux liés :
- au produit,
- à l’utilisateur,
- et au contexte de commande

peuvent être combinés pour construire un pipeline prédictif exploitable dans un cas réel de recommandation ou de fidélisation client.

---

## Partie Dashboard Streamlit

En complément du notebook principal, un **dashboard Streamlit** a été développé afin de proposer une visualisation plus interactive et plus claire du projet.

### Objectifs du dashboard
Le dashboard permet de :

- visualiser les principaux résultats de l’EDA
- explorer plusieurs insights business
- afficher l’analyse des produits fréquemment achetés ensemble
- exécuter la partie machine learning de façon interactive
- comparer les performances des modèles directement dans l’interface

### Contenu du dashboard
Le dashboard contient plusieurs sections :

- **Overview** : aperçu global du projet et des données
- **EDA** : visualisations sur les produits, départements, heures et jours de commande
- **Business Insights** : produits les plus souvent ajoutés en premier, co-purchase pairs, idées de bundles
- **Machine Learning** : exécution des modèles, tableau de comparaison, métriques et interprétation
- **Data Quality** : vérifications simples sur les valeurs manquantes, doublons et distribution de la cible

### Remarque sur les performances
Le dataset étant volumineux, certaines parties du dashboard utilisent :
- de l’échantillonnage,
- des structures allégées,
- et des traitements optimisés

afin de garantir une exécution stable sur notre environnement.


### Lancer le dashboard
Depuis la racine du projet :

streamlit run app.py

## Dépendances nécessaires 

Les principales bibliothèques nécéssaires pour cette partie sont :
- pandas
- matplotlib
- scikit-learn
- streamlit

## Intérêt du dashboard

Le dashboard constitue un bonus intéressant du projet car il permet de :
- présenter les résultats de manière plus professionnelle 
- faciliter l'exploration des données 
- rendre la partie analytique et prédicitive plus lisible
- appuyer la présentation avec un support visuel interactif

