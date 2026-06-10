# Blood Donation Prediction

Predicting whether a donor will give blood in the next campaign, so blood banks can target outreach and forecast supply.

**Domain:** Healthcare / CRM analytics · **Task:** Binary classification on historical donation behaviour (recency, frequency, volume, tenure — RFM-style features)

## Approach

- **EDA** — donation recency/frequency distributions, target imbalance, feature correlations.
- **Modeling** — Logistic Regression, Random Forest, SVM and KNN benchmarked under one protocol.
- **Metric choice** — model selection on **ROC AUC** rather than raw accuracy, since the class distribution is imbalanced and ranking donors by likelihood is the actual business use-case.

## Results

| Model | Accuracy | ROC AUC |
| --- | --- | --- |
| **Logistic Regression** | 0.759 | **0.785** |
| SVM | 0.759 | 0.773 |
| KNN | 0.767 | 0.771 |
| Random Forest | 0.724 | 0.736 |

**Selected model: Logistic Regression** — best ROC AUC, fully interpretable coefficients (a feature, not a compromise, when the output drives human outreach decisions).

## Run it

```bash
pip install pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook PRCP-1011-BloodDonaPred.ipynb
```

## Takeaways

Recency and frequency dominate prediction; simple linear models beat ensembles on this dataset size; choosing the right evaluation metric mattered more than model complexity.
