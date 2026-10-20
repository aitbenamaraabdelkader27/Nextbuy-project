# NextBuy — Grocery Basket Intelligence

NextBuy turns grocery transaction tables into an interactive Streamlit dashboard for exploring purchase patterns, first-in-cart behaviour and product associations. Built from an EPITECH team project, the restoration adds validated data loading, reproducible modelling and a clearly documented evaluation of repeat purchase status. A public demonstration can run entirely on generated data, while the supplied Instacart extract remains available for local analysis.

**Technologies:** Python 3.12, pandas, NumPy, scikit-learn, Streamlit, Plotly, pytest, Playwright, Ruff, Jupyter.

**Features:** five analytical pages; basket-preserving filters; product exploration and CSV exports; exact pair mining with support, confidence and lift; baseline/logistic regression/random forest comparison; ROC and precision–recall curves; held-out article inspection; data-quality and methodology views.

**Skills demonstrated:** relational data validation, streamed cohort preparation, exploratory data analysis, market basket analytics, leakage prevention through separate history data, reproducible classification pipelines, metric interpretation, interactive dashboard design, automated application/browser testing and deployment preparation.

**Verified real-cohort evidence:** 20,256 observed orders, 202,199 article lines. Fixed diagnostic evaluation: random forest ROC-AUC 0.7484, average precision 0.8360; baseline ROC-AUC 0.5000, average precision 0.6391. These are retrospective repeat-status metrics, not next-basket forecasting results.

**Attribution:** original academic work completed as a team; contributor records remain in the historical documents. Avoid claiming sole authorship, a production recommender, prediction of future purchases, promotion uplift or a live deployment before it exists.

**French portfolio variant:** NextBuy transforme des données de commandes alimentaires en un dashboard Streamlit interactif pour explorer les habitudes d’achat et les associations de produits. La restauration du projet EPITECH apporte un chargement validé, des analyses de paniers et une comparaison reproductible de modèles classant le statut de réachat des articles déjà commandés. La démonstration publique peut fonctionner sur des données synthétiques, clairement distinguées des analyses locales Instacart.
