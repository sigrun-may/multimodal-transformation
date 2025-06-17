import pandas as pd
import optuna
import os
from sklearn.model_selection import KFold

os.environ["R_HOME"] = r"C:\Program Files\R\R-4.4.2"
from rpy2.robjects import pandas2ri, r, globalenv
from rpy2.robjects.packages import importr

pandas2ri.activate()

# ---- Load example data ----
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# apply cross-validation
for train_index, test_index in KFold(n_splits=5).split(df):
    train, test = df.iloc[train_index], df.iloc[test_index]

    # train model


# ---- Send data to R ----
globalenv["df"] = pandas2ri.py2rpy(df)

# ---- Load R libraries ----
importr("partykit")

# ---- R: Define evaluation function for tuning ----
r(
    """
library(partykit)

evaluate_ctree <- function(mincriterion, minsplit, minbucket, maxdepth, testtype) {
  df$target <- as.factor(df$target)
    
  ctrl <- partykit::ctree_control(
      mincriterion = mincriterion,
      minsplit = minsplit,
      minbucket = minbucket,
      maxdepth = maxdepth,
      testtype = testtype
    )

  model <- partykit::ctree(target ~ ., data = df, control = ctrl)
  preds <- predict(model)

  truth <- df$target
  cm <- table(truth, preds)

  sensitivity <- cm[2, 2] / sum(cm[2, ])
  specificity <- cm[1, 1] / sum(cm[1, ])
  return(mean(c(sensitivity, specificity)))  # balanced accuracy
}
"""
)


# ---- Optuna objective (returns only OOB score) ----
def objective(trial):
    mincriterion = trial.suggest_float("mincriterion", 0.80, 0.99)
    minsplit = trial.suggest_int("minsplit", 5, 50)
    minbucket = trial.suggest_int("minbucket", 2, 20)
    maxdepth = trial.suggest_int("maxdepth", 2, 6)
    testtype = trial.suggest_categorical("testtype", ["Univariate", "Bonferroni"])

    score = r["evaluate_ctree"](
        mincriterion=mincriterion,
        minsplit=minsplit,
        minbucket=minbucket,
        maxdepth=maxdepth,
        testtype=testtype,
    )[0]
    return score


# ---- Run tuning with Optuna ----
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

best_params = study.best_params
best_score = study.best_value
print("\n✅ Best hyperparameters:")
print(best_params)
print(f"✅ Best OOB balanced accuracy: {best_score:.4f}")

# ---- R: Extract selected features from best model ----
globalenv["mincriterion"] = best_params["mincriterion"]
globalenv["minsplit"] = best_params["minsplit"]
globalenv["minbucket"] = best_params["minbucket"]
globalenv["maxdepth"] = best_params["maxdepth"]
globalenv["testtype"] = best_params["testtype"]

r(
    """
train_and_get_features <- function(mincriterion, minsplit, minbucket, maxdepth, testtype) {
  df$target <- as.factor(df$target)
  ctrl <- ctree_control(mincriterion = mincriterion,
                          minsplit = minsplit,
                          minbucket = minbucket,
                          maxdepth = maxdepth,
                          testtype = testtype)

  model <- ctree(target ~ ., data = df, control = ctrl)
  importance <- varimp(model)
  print(length(importance))
    
  # Optional: sort and view top features
  importance_sorted <- sort(importance, decreasing = TRUE)
  print(head(importance_sorted, 10))
  return(importance)
}
"""
)

# ---- Get selected features back into Python ----
selected_features = list(
    r["train_and_get_features"](
        best_params["mincriterion"],
        best_params["minsplit"],
        best_params["minbucket"],
        best_params["maxdepth"],
        best_params["testtype"],
    )
)

print(f"\n✅ Selected features from best model: {selected_features}")
