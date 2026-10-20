
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
