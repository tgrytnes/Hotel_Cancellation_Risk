# Hotel Cancellation Risk (HCR): Leak-Aware Prediction at Booking Time

**Status: Project Complete ✓**

A machine learning system for predicting hotel booking cancellations at the time of reservation, achieving 91.8% ROC-AUC through careful feature engineering and temporal validation.

---

## Project Overview

This project develops a production-ready cancellation prediction system using 119,384 hotel bookings from two Portuguese hotels (2015-2017). By predicting cancellation probability at booking time, hotels can optimize overbooking policies, target retention campaigns, and implement risk-based deposit requirements.

### Key Results

- **Best Model**: Random Forest (100 estimators)
- **Performance**: ROC-AUC 0.917 | PR-AUC 0.895 | Precision 82.3% | Recall 75.8%
- **Improvement**: 9% ROC-AUC gain over logistic regression baseline
- **Top Predictors**: Lead time (0.203), ADR/pricing (0.195), customer engagement signals

---

## Dataset

**Hotel Booking Demand Dataset** from Antonio, Almeida, and Nunes (2019), containing:
- 119,390 bookings across city and resort hotels
- 32 variables including lead time, market segment, distribution channel, guest history
- 37% cancellation rate (moderate class imbalance)
- Temporal span: July 2015 - August 2017

**Source**: [Kaggle - Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

---

## Methodology

### Data Preprocessing

**Leakage Prevention**:
- Removed `reservation_status`, `reservation_status_date`, `deposit_type` (post-outcome information)
- Removed 7 bookings with undefined market segments/channels (<0.006%)

**Missing Value Treatment**:
- `agent`, `company`: Filled with 0 (represents direct bookings)
- `children`: Filled with 0 (logical default)
- `country`: Preserved as "Unknown" category (13.7% cancellation rate segment)

**Outlier Handling**:
- Removed 1 negative ADR booking
- Preserved zero-price bookings with actual stays (legitimate complimentary segment)
- Capped extreme ADR outlier at 99.99th percentile

### Feature Engineering

**50 modeling features** including:
- Raw signals: lead_time, adr, guest counts, stay duration
- Engineered aggregations: total_stay_nights (weekend + weekday)
- Lead time buckets: categorical risk thresholds (Last Minute, Short, Medium, Long)
- Interaction terms: market_segment × distribution_channel combinations

### Model Development

**Three-Model Strategy**:

1. **Logistic Regression** (baseline): Interpretable linear model with L1/L2 regularization, class balancing, and standardized features
2. **Random Forest**: Stable bagging ensemble (100-300 trees) with class weighting
3. **XGBoost**: Gradient boosting with hyperparameter tuning (learning rate, max depth) and scale_pos_weight for imbalance

**Training Setup**:
- Chronological split: 75% train / 25% test (time-aware validation)
- 5-fold stratified cross-validation for hyperparameter tuning
- Random seed 42 for reproducibility
- Primary metric: PR-AUC (class imbalance aware)
- Secondary metrics: ROC-AUC, Brier Score

---

## Results

### Model Performance Comparison

| Model | ROC-AUC | PR-AUC | Precision | Recall | Accuracy |
|-------|---------|--------|-----------|--------|----------|
| **Random Forest (100)** | 0.917 | 0.895 | 82.3% | 75.8% | 84.9% |
| Random Forest (300) | 0.918 | 0.895 | 82.3% | 75.8% | 85.0% |
| XGBoost (lr=0.1, d=5) | 0.888 | 0.852 | 73.6% | 75.1% | 80.8% |
| Logistic Regression | 0.841 | 0.797 | 62.8% | 76.6% | 74.6% |

### Statistical Validation

- **RF vs XGBoost**: t=4.032, p=0.0050 (Significant)
- **RF vs LogReg**: t=386.6, p<0.0001 (Highly Significant)
- Performance differences are statistically robust across cross-validation folds

### Feature Importance (Random Forest)

**Top 5 Predictors**:
1. Lead time: 0.203 (booking horizon)
2. ADR (Average Daily Rate): 0.195 (pricing)
3. Total special requests: 0.081 (engagement)
4. Agent: 0.067 (distribution channel)
5. Total stay nights: 0.044 (commitment signal)

**Key Insight**: Lead time and pricing dominate predictions (40% combined importance), while customer engagement signals (special requests, parking, stay duration) form the second tier (19%).

---

## Production Recommendation

**Deploy Random Forest with 100 estimators**

**Rationale**:
- Performance nearly identical to 300-tree variant (0.917 vs 0.918 ROC-AUC)
- 3x faster training time
- 67% smaller model size
- Better trade-off for production constraints (retraining frequency, inference speed, infrastructure costs)

**Business Impact**:
- Correctly ranks 92% of cancellations higher than non-cancellations
- 64% reduction in false alarms vs logistic regression (1,807 vs 5,007 FP)
- For 30,000 annual bookings: ~3,200 fewer unnecessary interventions

---

## Limitations

1. **Generalizability**: Dataset limited to Portuguese hotels (2015-2017); may not transfer to other markets or post-COVID patterns
2. **Feature Scope**: Restricted to booking-time information; excludes external signals (reviews, events, competitor pricing)
3. **Temporal Coverage**: Only 2 years of data insufficient for robust seasonal pattern detection
4. **Causal Inference**: Models identify associations, not causal effects of interventions (deposits, confirmations)

---

## Next Steps

### Immediate Deployment
- Implement REST API for real-time scoring
- Set up drift monitoring dashboard (feature distributions, performance metrics)
- Conduct A/B test against current cancellation policies

### Future Enhancements
- Incorporate external data sources (local events, online reviews, dynamic pricing)
- Expand to multi-property and multi-market validation
- Develop causal inference framework for intervention testing (do deposits actually reduce cancellations?)
- Implement automated retraining pipeline with concept drift detection

---

## Repository Structure

```
.
├── artifacts/                  # Model outputs, metrics, predictions
│   ├── results.csv             # All experiment results
│   ├── features.csv            # Engineered feature table
│   └── artifacts_*/            # Per-experiment outputs (metrics.json, predictions.json, etc.)
├── configs/                    # Experiment configurations (YAML)
├── data/
│   └── raw/hotel_bookings.csv  # Source dataset
├── notebooks/
│   └── main.ipynb              # Complete analysis notebook (EDA, training, evaluation)
├── src/hcr/                    # Source code package
│   ├── features.py             # Feature engineering pipeline
│   ├── train.py                # Model training script
│   └── ...
├── reports/                    # Generated reports and visualizations
├── scripts/
│   ├── run_train.sh            # Training pipeline entrypoint
│   └── init_project.py         # Project initialization script
├── pyproject.toml              # Package configuration
└── README.md                   # This file
```

---

## Reproducibility

All experiments use **random seed 42** for reproducibility. To replicate results:

```bash
# 1. Set up environment
source bootstrap_env.sh

# 2. Run feature engineering
python src/hcr/features.py

# 3. Train models
bash scripts/run_train.sh configs/exp_baseline.yaml

# 4. View results
cat artifacts/results.csv
```

---

## References

Antonio, N., Almeida, A., & Nunes, L. (2019). Hotel Booking Demand Datasets. *Data in Brief*, 22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126

Kaggle dataset mirror: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

---

## License

This project is distributed under the license specified in `LICENSE`.

---

## Contact

For questions or collaboration inquiries, please open an issue in this repository.