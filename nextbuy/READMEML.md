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

