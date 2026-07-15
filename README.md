# Cat Breed Classifier

Progetto di classificazione supervisionata per predire la **razza** di un gatto
(`razza`) a partire da caratteristiche fisiche e comportamentali, per la Classification
Challenge "Cat Breed Classifier".

## Struttura della repository

```
.
├── cat_breed_classifier.ipynb   # Notebook con l'intera pipeline (EDA, cleaning,
│                                 # training, tuning, valutazione, predizioni)
├── data/
│   ├── cats_dataset.csv         # Training set fornito
│   └── test_set.csv             # Test set fornito (da predire)
├── grafici/
│   ├── 01_distribuzione_razze.png
│   ├── 02_heatmap_correlazione.png
│   ├── 03_peso_per_razza.png
│   └── 04_matrice_confusione.png
├── predictions.csv              # Predizioni finali sul test set (ID, razza_prevista)
└── README.md
```

## Come riprodurre i risultati

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
jupyter nbconvert --to notebook --execute --inplace cat_breed_classifier.ipynb
```

Il notebook rigenera automaticamente tutti i grafici nella cartella `grafici/` e il
file `predictions.csv`.

## Sintesi del lavoro svolto

1. **EDA**: analisi di dimensioni, tipi, valori mancanti e distribuzione del target.
   Il training set fornito contiene diverse anomalie realistiche: una riga vuota, una
   riga con colonne sfasate (dato corrotto), formattazioni incoerenti nelle variabili
   categoriche (maiuscole/minuscole, spazi vs underscore, sinonimi) e una classe target
   spuria (`Alien`, 5 record) non rappresentativa del problema.
2. **Cleaning**: rimozione di righe corrotte/vuote e della classe outlier, normalizzazione
   del testo, unificazione dei sinonimi, invalidazione dei valori fuori dominio (probabili
   "leak" da altre colonne) e imputazione (mediana/moda).
3. **Modeling**: confronto in cross-validation tra Logistic Regression, Random Forest e
   XGBoost; selezione automatica del modello con F1-macro più alto e ottimizzazione degli
   iperparametri via `GridSearchCV`.
4. **Valutazione**: accuracy, F1-macro, classification report e matrice di confusione sul
   validation set (accuracy ≈ 0.97).
5. **Predizioni**: il modello finale, riaddestrato su tutto il training set pulito, genera
   `predictions.csv` con le razze previste per i 50 gatti del test set.
