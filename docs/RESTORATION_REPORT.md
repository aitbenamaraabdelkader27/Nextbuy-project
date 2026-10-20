# NextBuy — rapport de restauration

## 1. Projet original

Le code et les deux notebooks identifient un projet EPITECH d’exploration de commandes alimentaires issues d’un extrait Instacart 2017. Il étudie les produits et départements, les horaires, les tailles de paniers, les articles ajoutés en premier, les paires de co-achats et la classification de `reordered` par régression logistique et forêt aléatoire. Il ne contient pas de moteur de recommandation opérationnel ni de vraie prédiction de prochain panier.

La cible indique qu’un article déjà acheté avait été commandé auparavant par ce client. L’interface initiale comportait cinq onglets, des échantillons de lignes et un bouton d’entraînement. Les métriques initiales étaient ROC-AUC, Average Precision et classification report. Le notebook enregistrait notamment un ROC-AUC de 0,7714 pour la forêt ; cette ancienne valeur dépend d’un autre protocole et n’est pas présentée comme performance validée de la restauration.

Le dépôt contient cinq CSV : 1 444 444 commandes, 13 692 886 lignes d’articles, 49 688 produits, 134 rayons et 21 départements. Les documents README, READMEEDA, READMEML, READMENM, insights, le notebook principal et son checkpoint ont été lus. Aucune recette d’extraction, licence de données, prix, date calendaire ni modèle sauvegardé n’a été fourni.

## 2. Modifications

Corrections principales : chemins dépendant du répertoire courant, fusion complète en mémoire avant échantillonnage, paniers fragmentés par tirage de lignes, confusion entre commandes et articles dans les histogrammes horaires, labels manquants, absence de baseline, agrégats d’entraînement utilisant leur propre cible, prétraitement partagé entre deux pipelines et absence de normalisation logistique. Le notebook historique rechargeait aussi plusieurs CSV avec un séparateur incorrect et son checkpoint importait des fonctions inexistantes. Un notebook restauré exécutable remplace ce parcours pour les démonstrations, sans effacer les originaux.

L’architecture finale sépare `nextbuy_core/data.py`, `analytics.py`, `modeling.py`, `charts.py` et `ui.py`. `app.py` est l’entrée principale ; `nextbuy/app.py` est un lanceur compatible. L’ancien dashboard et son module ML sont copiés à l’identique dans `docs/archive/`. Le module académique `nextbuy/src/modeling.py` reste une référence historique, non utilisée par l’application restaurée.

Fichiers créés : entrée racine, modules applicatifs, quatre scripts (`prepare_data`, `audit_data`, `evaluate`, `browser_check`), quatre fichiers de tests, notebook exécuté, CSV synthétiques et manifeste, configuration Streamlit, `.python-version`, `.gitignore` racine, `pyproject.toml`, dépendances directes/dev/verrouillées, README anglais et documents de provenance, déploiement, portfolio et validation. Les rapports JSON et captures sont enregistrés dans `docs/`. Fichiers modifiés : lanceur et dépendances historiques, note d’orientation dans `nextbuy/README.md`. Les CSV sources et le notebook original sont préservés.

## 3. Data Science

L’audit intégral est disponible dans `source_audit.json`. Il constate notamment 31 identifiants clients manquants, 449 identifiants produits manquants, 446 positions de panier manquantes, 449 cibles manquantes et 86 903 commandes sans lignes. Aucun doublon de ligne ou de clé n’est relevé dans les cinq fichiers sources. Les intervalles au-delà de la première commande comportent aussi des nulls invalides. La position de panier maximale est 127 ; les grandes valeurs valides ne sont pas arbitrairement tronquées.

La préparation exclut 129 commandes sources au contexte invalide. Un tirage uniforme de 3 000 identifiants clients (seed 42) conserve toutes les lignes disponibles de leurs commandes ; 16 paniers de cette cohorte sont exclus pour corruption ou positions non contiguës. Les commandes sans lignes sont écartées. Résultat : 2 948 clients observés, 20 256 commandes et 202 199 lignes. Le manifeste local enregistre les exclusions et les empreintes SHA-256. Les clés, types, plages, relations et positions sont validés avant fusion. La cible n’est jamais imputée.

Indicateurs réels : part d’articles déjà achetés **58,87 %**, taille moyenne observée **9,98 articles** par panier. Les co-achats sont calculés par énumération exacte des paires parmi les produits candidats les plus fréquents. Support, confiance directionnelle et lift utilisent des dénominateurs explicites, avec toutes les commandes sélectionnées dans le support.

Le protocole ML réserve les commandes anciennes à l’historique, l’avant-dernière observée à l’entraînement et la dernière au test. Il exclut 890 clients ayant moins de trois paniers. Historique : 145 303 lignes ; entraînement : 21 010 ; test : 21 729. Les agrégats lissés sont appris exclusivement dans l’historique puis figés. Les variables numériques sont standardisées et les catégories encodées avec prise en charge des catégories inconnues. Les paramètres sont fixés, les pipelines indépendants et la seed égale à 42. Aucun réglage ni choix automatique du gagnant n’utilise le test.

| Modèle | ROC-AUC | Average Precision | Brier |
| --- | ---: | ---: | ---: |
| Prévalence constante | 0,500000 | 0,639054 | 0,231467 |
| Régression logistique | 0,737694 | 0,825968 | 0,196577 |
| Forêt aléatoire | 0,748398 | 0,836000 | 0,194595 |

Limites : extrait incomplet et population modifiée par le nettoyage ; ordre relatif par client sans chronologie calendaire globale ; sélection des seuls produits achetés ; position dans le panier connue après sélection ; dernier panier observé potentiellement différent du dernier panier réel ; absence de validation externe, d’intervalles de confiance et d’étude de calibration. Aucune causalité promotionnelle ni performance future n’est affirmée. Les résultats synthétiques sont séparés des résultats réels dans `evaluation.json`.

## 4. Dashboard

Pages finales : **Overview**, **Data explorer**, **Basket insights**, **Model lab**, **Methodology & quality**.

Interactions : sélection de la source ; filtres de jours codés et d’heures ; département pour l’analyse d’articles ; choix des distributions ; recherche littérale ; exports CSV ; seuil d’observations ; nombre de produits candidats ; minimum de co-achats et classement des paires ; exécution explicite de l’évaluation ; inspection d’un modèle, de son rapport, de sa matrice et d’un article de test.

La direction visuelle utilise un fond clair, une palette teal/indigo/ambre, une sidebar identifiée, quatre cartes KPI, des conteneurs natifs, des graphiques Plotly harmonisés et des explications courtes. Le CSS ne cible que les classes du balisage créé par l’application. Les états vides, chargements et erreurs de données sont explicites. Les données et expériences sont mises en cache avec deux entrées maximum ; aucun entraînement n’a lieu au simple chargement d’une page.

Chromium a ouvert les cinq pages, vérifié leurs graphiques, lancé les modèles, changé un article inspecté et téléchargé un CSV de recherche. Les captures ont été examinées sur ordinateur, tablette et mobile ; pas de débordement horizontal du document aux deux petites tailles. Les rapports détaillés sont dans `browser-check.json` et `VALIDATION.md`.

## 5. Tests

**15 tests passent**, couvrant chargement, validation, exclusions de paniers corrompus, statistiques, filtres, dénominateurs des paires, séparation des partitions, absence d’utilisation des labels entraînement/test dans les features, catégories inconnues, baseline, reproductibilité et navigation Streamlit.

Ruff et `pip check` passent. L’évaluation réelle et synthétique a été exécutée, puis reproduite dans un environnement vierge. Une copie publique sans CSV réels passe les cinq pages et le calcul des modèles dans cet environnement. Les quatre cellules Python du notebook restauré sont exécutées sans erreur. Le serveur Streamlit et Chromium ont réellement fonctionné. Les limites de portée et les erreurs détectées puis corrigées sont consignées dans `VALIDATION.md`.

## 6. Exécution

Depuis `/home/abdelkader/Nextbuy-project` :

```bash
.venv/bin/python -m streamlit run app.py
```

Environnement vérifié : **Python 3.12.3**. Les dépendances principales sont NumPy, pandas, Plotly, scikit-learn et Streamlit, épinglées dans `requirements.txt`. L’environnement complet d’installation propre est dans `requirements.lock.txt`. Les outils de test et de notebook sont séparés dans `requirements-dev.txt`.

## 7. Déploiement et Git

L’entrée **`app.py`**, la configuration, les dépendances et les petites données synthétiques sont prêtes pour Streamlit Community Cloud. Aucun secret, compte tiers, modèle binaire ou téléchargement à l’exécution n’est requis. La version Python 3.12 doit être choisie dans les paramètres du déploiement. L’installation propre et le scénario sans données réelles ont été vérifiés localement ; le service hébergé, ses quotas et son démarrage à froid restent à vérifier après autorisation.

Les données brutes font environ 270 MiB, dont un CSV de 233 MiB ; elles sont ignorées par Git, comme la cohorte réelle dérivée et l’environnement virtuel. Le mode synthétique représente environ 0,6 Mo de CSV. Les licences des fichiers réels et des travaux de groupe doivent être confirmées avant leur redistribution ; cela n’empêche pas le fonctionnement de la démonstration synthétique indépendante.

`origin` pointe bien vers `git@github.com:aitbenamaraabdelkader27/Nextbuy-project.git`. Aucun historique EPITECH n’a été copié, aucun commit/push ni changement de visibilité effectué. La recherche de motifs courants de secrets dans les fichiers applicatifs/documentaires n’a pas trouvé de correspondance ; elle ne constitue pas une certification exhaustive.

## 8. Portfolio

**Titre :** NextBuy — Grocery Basket Intelligence.

NextBuy transforme des commandes alimentaires en un dashboard interactif pour explorer les comportements d’achat, les premiers articles du panier et les associations de produits. La restauration du projet EPITECH ajoute une validation des données, un protocole reproductible de classification du statut de réachat et une démonstration synthétique indépendante des fichiers réels.

**Stack :** Python, pandas, NumPy, scikit-learn, Streamlit, Plotly, pytest, Playwright, Ruff, Jupyter.

**À valoriser :** données relationnelles, préparation par lecture en blocs, analytics de paniers, maîtrise des fuites de labels, interprétation des métriques, modularité, interface interactive, tests applicatifs et préparation cloud. Le README anglais et `PORTFOLIO.md` proposent une version directement réutilisable, en conservant l’attribution au travail collectif initial.
