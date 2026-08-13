# Store Sales Time Series Forecasting — prototype audité

Exercice d'apprentissage sur une compétition passée. **Les résultats numériques du premier passage
ne constituent pas un benchmark valide** et ne doivent pas être présentés comme un score Kaggle.

## Verdict de l'audit (GPT-5.6 Sol, 13 août 2026)

1. La métrique officielle est **RMSLE**, pas SMAPE.
2. Les rolling target features étaient calculées sans décalage préalable :
   `rolling(...).mean()` incluait la vente du jour prédit.
3. Les moyennes cible par store/family étaient calculées sur l'ensemble des données avant la CV,
   donc elles incorporaient les cibles de validation.
4. Les 16 jours futurs exigent une stratégie multi-horizon explicite (recursive, directe par
   horizon ou multi-output). Concaténer train/test ne suffit pas à produire les lags futurs.

Les anciens résultats SMAPE (~0.45–0.53) sont donc conservés uniquement comme **incident pédagogique** :
ils ne mesurent ni la métrique officielle ni une validation causale.

## Ce qui reste utile

- EDA de la structure : 54 magasins × 33 familles, horizon de 16 jours.
- Fusion des covariables et construction du pipeline de données.
- Découverte pratique du walk-forward et des risques de fuite temporelle.
- Leçon centrale : vérifier la métrique officielle et auditer la causalité de chaque feature avant
  tout entraînement coûteux.

## Reconstruction requise

1. Implémenter RMSLE et deux baselines causales : seasonal-naive et moyenne saisonnière.
2. Construire des lags/rolling avec `shift(1)` ou `shift(h)`.
3. Évaluer sur au moins 3 fenêtres de 16 jours ; reporter moyenne, écart-type et pire fenêtre.
4. Tester un LightGBM sur `log1p(sales)` avec stratégie directe par horizon ou recursive.
5. Produire une submission réelle et reproductible avant de réintégrer Store Sales aux résultats
   validés du portfolio.

## Fichiers

Les scripts `code/eda.py`, `improved.py`, `v3_transform.py`, `v5_walk.py` sont archivés comme
prototypes historiques. Ils ne doivent pas être réutilisés sans correction du leakage et de la
métrique.
