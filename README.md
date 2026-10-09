# Titanic Survival Classification

Predicting which Titanic passengers survived with four classification models, comparing how well each one generalizes after hyperparameter tuning. Built during the Women in AI Fellowship 2026 (Webfala Digital Skills for All Initiative).

The project focuses on **overfitting**: how limiting model complexity, cross-validation and precision/recall change how far a model's score can be trusted.

## Dataset

887 passengers. Target: `Survived` (0 = died, 1 = survived). Features: `Pclass`, `Sex`, `Age`, `SibSp`, `PaCh`, `Fare`. `Name` was dropped and `Sex` encoded (female = 0, male = 1).

## Approach

1. Dropped `Name`, encoded `Sex`, separated features and target
2. Split 70/30 (267 test passengers) with `stratify=y` and `random_state=20`
3. Put Logistic Regression and SVM in a `Pipeline` with `StandardScaler`, so scaling is fitted on training data only (trees and forests don't need it)
4. Tuned with `GridSearchCV` (5-fold CV, training set only), using the test set once per model at the end
5. Evaluated with accuracy, confusion matrix, precision, recall, F1, and train vs test accuracy

## Results

Test accuracy is on the 267 held-out passengers. CV is 5-fold cross-validation on the training set.

| Model               | Settings                                | CV accuracy | Test accuracy |
| ------------------- | --------------------------------------- | ----------: | ------------: |
| Logistic Regression | default                                 |         n/a |         77.9% |
| Logistic Regression | tuned (`C=10`, liblinear)               |       81.8% |         77.9% |
| Decision Tree       | `max_depth=3`                           |         n/a |         83.1% |
| Decision Tree       | tuned (`max_depth=5`, `min_samples_leaf=3`) |     82.6% |         76.8% |
| Random Forest       | `max_depth=5`, 100 trees                |         n/a |         82.8% |
| Random Forest       | tuned (`max_depth=7`, 100 trees)        |       82.4% |         81.6% |
| SVM                 | default                                 |         n/a |         80.9% |
| SVM                 | tuned (`C=10`, `gamma=0.1`, rbf)        |       83.7% |         77.9% |

**Overfitting check (train vs test):** Random Forest (`max_depth=5`) 87.9% vs 82.8%; SVM (default) 84.7% vs 80.9%.

**Final model:** the tuned Random Forest, because its CV and test scores agree within one point. Confusion matrix (rows = actual, columns = predicted):

|                    | Predicted died | Predicted survived |
| ------------------ | -------------: | -----------------: |
| **Actually died**     |            140 |                 24 |
| **Actually survived** |             25 |                 78 |

Precision and recall for the survivor class are both 0.76.

## Prediction Function

`predict_survival()` takes a passenger's details and returns a message with the tuned Random Forest's probabilities for died and survived. `Sex` uses the training encoding (0 = female, 1 = male).

```python
predict_survival("Tomi", Pclass=1, Sex=1, Age=25.0, SibSp=0, PaCh=0, Fare=100)
```

## Limitations

- Small dataset and a single train/test split, so scores carry noticeable noise
- The depth-3 tree and depth-5 forest were hand-picked after seeing test results, so their scores are slightly optimistic
- No feature engineering beyond encoding `Sex` (for example, no title extraction or family-size features)

## How to Run

```bash
git clone https://github.com/salamatsaheed-ux/Titanic-Survival-Classification.git
cd Titanic-Survival-Classification
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook Titanic-classification.ipynb
```

Place the dataset at `data/titanic.csv` first, then run the notebook from top to bottom.

**Tools:** Python · Pandas · NumPy · Matplotlib · Scikit-learn · Jupyter Notebook

## Author

**Salamat Saheed Togun**
