import time

import numpy as np
import pandas as pd
import optuna
import os

from optuna.samplers import TPESampler
from sklearn.metrics import f1_score, log_loss, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from multimodal_transformation import multimodal_transformation as mt
# from ucimlrepo import fetch_ucirepo

os.environ["R_HOME"] = r"C:\Program Files\R\R-4.4.2"
from rpy2.robjects import pandas2ri, r, globalenv, FloatVector
from rpy2.robjects.packages import importr

pandas2ri.activate()

# ---- Load R libraries ----
importr("partykit")

# ---- R: Define evaluation function for tuning ----
r(
    """
        library(partykit)

        evaluate_ctree <- function(mincriterion, minsplit, minbucket, maxdepth, testtype) {
          train$Label <- as.factor(train$Label)

          ctrl <- partykit::ctree_control(
              mincriterion = mincriterion,
              minsplit = minsplit,
              minbucket = minbucket,
              maxdepth = maxdepth,
              testtype = testtype
            )

          model <- partykit::ctree(Label ~ ., data = train, control = ctrl)
          predicted_labels <- predict(model, newdata = test)
          predicted_labels_prob <- predict(model, newdata = test, type = "prob")

          # truth <- test$Label
          # cm <- table(truth, preds)
          # 
          # # print(truth)
          # # print(preds)
          # # print(cm)
          # 
          # sensitivity <- cm[2, 2] / sum(cm[2, ])
          # specificity <- cm[1, 1] / sum(cm[1, ])
          # return(mean(c(sensitivity, specificity)))  # balanced accuracy
          
          # list of true labels, predicted labels, and predicted probabilities
          result_list <- c(
            Label = as.factor(test$Label),
            Predicted = predicted_labels,
            Predicted_Probabilities = predicted_labels_prob
          )
          # send result list to python
          return(result_list)
          # return (list(test$Label, predicted_labels, predicted_labels_prob))
        }
        """
)

r(
    """
train_and_get_features <- function(mincriterion, minsplit, minbucket, maxdepth, testtype) {
  df$Label <- as.factor(df$Label)
  ctrl <- ctree_control(mincriterion = mincriterion,
                          minsplit = minsplit,
                          minbucket = minbucket,
                          maxdepth = maxdepth,
                          testtype = testtype)

  model <- ctree(Label ~ ., data = df, control = ctrl)
  importance <- varimp(model)
  print(length(importance))

  # Optional: sort and view top features
  importance_sorted <- sort(importance, decreasing = TRUE)
  print(head(importance_sorted, 10))
  return(importance)
}
"""
)

# ---- Load example data ----
from sklearn.datasets import load_breast_cancer
# from mltb2.data import load_colon
# import platformdirs

# set start time as utc now
start_time = time.time()

# # fetch dataset
# hcv_data = fetch_ucirepo(id=571)
#
# label_series = hcv_data.data.targets["Category"]
# data_df = hcv_data.data.features
# data_df.drop(columns=["Age","Sex"], inplace=True)
#
# # metadata
# print(hcv_data.metadata)
#
# # variable information
# print(hcv_data.variables)

# # load colon cancer data
# label_series, data_df = load_colon()
# # convert integer column names to string
# data_df.columns = [f"f_{i}" for i in range(data_df.shape[1])]
#
# # convert labels to string
# label_series = [str(label) for label in label_series]
# data_df.insert(loc=0, column="Label", value=label_series)


# read z-log csv
data_df = pd.read_csv("data/hcvdat0_zlog.csv")
data_df.insert(loc=0, column="Label", value=data_df["Category"])
data_df.drop(columns=["Category", "X", "Age","Sex"], axis=1, inplace=True)

# remove all samples with suspect Blood Donor als Label
data_df = data_df[data_df["Label"] != "0s=suspect Blood Donor"]


testgrid_classes = ["1=Hepatitis", "2=Fibrosis", "3=Cirrhosis"]

# data_df = data_df[data_df["Label"] != "1=Hepatitis"]
# data_df = data_df[data_df["Label"] != "2=Fibrosis"]


# data_df.to_csv("../archive/hcv_df_z_log.csv")
# data_df = pd.read_csv("../archive/hcv_df.csv")

data_df_cp = data_df.copy()

# # # scale current feature
# scaler = MinMaxScaler()
# # scaler = RobustScaler()
# # scaler = StandardScaler()
# # scaled_values = scaler.fit_transform(data_df[data_df.columns[1:]].values)
# # data_df = pd.DataFrame(scaled_values, columns=data_df.columns[1:])
# # data_df["Label"] = data_df_cp["Label"].values
#
# data_df = mt.transform_data_set(data_df, interpolation_method="CubicSpline", target_overlap=0.001, scale_std=0.3,  plot=False, inplace=True)
#std_list=[0.3,0.3,0.3,0.3],
# reset index for cross validation splits

# data_df = data_df.reset_index(drop=True)

# unique_labels = data_df["Label"].unique()
# for feature in data_df.columns[1:]:
#     # if not "i" in feature and not "21" in feature:
#     #     continue
#
#     # calculate number of peaks
#     peaks = mt.count_peaks(data_df_cp[feature].values)
#
#     # calculate the effect size before and after the transformation
#     original_cohens_d = mt.cohens_d(
#         data_df_cp[data_df_cp["Label"] == unique_labels[0]][feature].values,
#         data_df_cp[data_df_cp["Label"] == unique_labels[1]][feature].values,
#     )
#     transformed_cohens_d = mt.cohens_d(
#         data_df[data_df["Label"] == unique_labels[0]][feature].values,
#         data_df[data_df["Label"] == unique_labels[1]][feature].values,
#     )
#     robust_original_cohens_d = mt.robust_cohens_d(
#         data_df_cp[data_df_cp["Label"] == unique_labels[0]][feature].values,
#         data_df_cp[data_df_cp["Label"] == unique_labels[1]][feature].values,
#     )
#     robust_transformed_cohens_d = mt.robust_cohens_d(
#         data_df[data_df["Label"] == unique_labels[0]][feature].values,
#         data_df[data_df["Label"] == unique_labels[1]][feature].values,
#     )
#     # print(feature)
#     # # print the effect size
#     # print(f"Effect size before transformation: {original_cohens_d:.4f}")
#     # print(f"Effect size after transformation: {transformed_cohens_d:.4f}")
#     # print(f"Robust effect size before transformation: {robust_original_cohens_d:.4f}")
#     # print(f"Robust effect size after transformation: {robust_transformed_cohens_d:.4f}")
#     #
#     # print(is_bimodal(transformed_data_df[feature].values, verbose=True))
#     if transformed_cohens_d < original_cohens_d:
#         print(f"{feature} Effect size was reduced")
#         print(f"Effect size before transformation: {original_cohens_d:.4f}")
#         print(f"Effect size after transformation: {transformed_cohens_d:.4f}")
#         print(f"Robust effect size before transformation: {robust_original_cohens_d:.4f}")
#         print(f"Robust effect size after transformation: {robust_transformed_cohens_d:.4f}")
#         print(mt.is_bimodal(data_df_cp[feature].values, verbose=True))
#         print(" ")
#     else:
#         print(f"{feature} Difference of robust Cohen's d: {robust_transformed_cohens_d - robust_original_cohens_d:.4f}")
#         print(f"{feature} Difference of Cohen's d: {transformed_cohens_d - original_cohens_d:.4f}")

# reset index for cross validation splits
data_df = data_df.reset_index(drop=True)
# df = data_df
# assert data_df.columns[0] == "label"
# data = load_breast_cancer()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df["Label"] = data.Label


def cross_validate(df, mincriterion, minsplit, minbucket, maxdepth, testtype):
    scores = []
    # apply cross-validation
    for train_index, test_index in StratifiedKFold(n_splits=5).split(df, df["Label"]):
        train, test = df.iloc[train_index], df.iloc[test_index]

        result_list = []

        # ---- Send data to R ----
        globalenv["train"] = pandas2ri.py2rpy(train)
        globalenv["test"] = pandas2ri.py2rpy(test)
        globalenv["result_list"] = FloatVector(result_list)

        # train model
        ctree_result = r["evaluate_ctree"](
            mincriterion=mincriterion,
            minsplit=minsplit,
            minbucket=minbucket,
            maxdepth=maxdepth,
            testtype=testtype,
        )
        y_true = ctree_result[:len(test.index)]
        y_predicted = ctree_result[len(test.index) : 2 * len(test.index)]
        y_predicted_probabilities = ctree_result[2 * len(test.index) :]

        # f1 = f1_score(y_true, y_predicted, average="weighted")
        f1 = f1_score(y_true, y_predicted)


        # select only class 1 and 2

        # y_true = y_true[(y_true != 3) & (y_true != 4)]
        # y_predicted = y_predicted[(y_predicted != 3) & (y_predicted != 4)]

        # print(confusion_matrix(y_true, y_predicted))
        # ConfusionMatrixDisplay(confusion_matrix(y_true, y_predicted))
        #
        # print(classification_report(y_true, y_predicted))

        # calculate score for class 1 versus 2 only
        # f1 = f1_score(y_true, y_predicted, average="binary", pos_label=2)

        # lloss = log_loss(y_true, y_predicted_probabilities)

        scores.append(f1)
    return np.mean(scores)


def optimize_c_trees(preprocessed_data_df):
    def objective(trial):
        mincriterion = trial.suggest_float("mincriterion", 0.40, 0.99)
        minsplit = trial.suggest_int("minsplit", 3, 50)
        minbucket = trial.suggest_int("minbucket", 2, 20)
        maxdepth = trial.suggest_int("maxdepth", 1, 6)
        # testtype = trial.suggest_categorical("testtype", ["Univariate", "Bonferroni"])
        testtype = "Bonferroni"
        return cross_validate(preprocessed_data_df, mincriterion, minsplit, minbucket, maxdepth, testtype)

    # ---- Run tuning with Optuna ----
    sampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=40)


    best_params = study.best_params
    best_score = study.best_value
    print("\n✅ Best hyperparameters:")
    print(best_params)
    print(f"✅ Best Score: {best_score:.4f}")

    # ---- R: Extract selected features from best model ----
    globalenv["mincriterion"] = best_params["mincriterion"]
    globalenv["minsplit"] = best_params["minsplit"]
    globalenv["minbucket"] = best_params["minbucket"]
    globalenv["maxdepth"] = best_params["maxdepth"]
    # globalenv["testtype"] = best_params["testtype"]
    globalenv["testtype"] = "Bonferroni"


    # ---- Get selected features back into Python ----
    mincriterion = best_params["mincriterion"]
    minsplit = best_params["minsplit"]
    minbucket = best_params["minbucket"]
    maxdepth = best_params["maxdepth"]
    testtype = "Bonferroni"

    # ---- Send data to R ----
    globalenv["df"] = pandas2ri.py2rpy(preprocessed_data_df)

    selected_features = list(
        r["train_and_get_features"](
            mincriterion=mincriterion,
            minsplit=minsplit,
            minbucket=minbucket,
            maxdepth=maxdepth,
            testtype=testtype,
        )
    )
    return best_score, selected_features

print("start cross-validation")
result_dict = {}

scale = ["min_max", "multimodal"]
# # scale current feature
scaler = MinMaxScaler()
# scaler = RobustScaler()
# scaler = StandardScaler()
# scaled_values = scaler.fit_transform(data_df[data_df.columns[1:]].values)
# data_df = pd.DataFrame(scaled_values, columns=data_df.columns[1:])
# data_df["Label"] = data_df_cp["Label"].values
for scaling in scale:
    result_dict[scaling] = {}

    for class_name in testgrid_classes:
        classes_to_drop = [c for c in testgrid_classes if c != class_name]
        print(f"Drop classes: {classes_to_drop}")
        df = data_df.copy()
        for class_to_drop in classes_to_drop:
            df = df[df["Label"] != class_to_drop]

        # change categorical labels to int
        df["Label"] = df["Label"].astype("category")
        df["Label"] = df["Label"].cat.codes
        assert df['Label'].unique()[1] == 1, "The labels are not 0 and 1. Please check the data."
        assert df["Label"].nunique() == 2, "There are more than 2 classes in the dataset. Please check the data."

        if scaling == "min_max":
            df[data_df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]].values)
        elif scaling == "multimodal":
            df = mt.transform_data_set(df, interpolation_method="CubicSpline", target_overlap=0.1,
                                            scale_std=1, plot=False, inplace=True)

        print(f"Cross-validation for class: {class_name}")
        result_dict[scaling][class_name] = optimize_c_trees(df)


# pretty print the result dict
import pprint
pprint.pprint(result_dict)

print(f"--- {time.time() - start_time} seconds ---")
print(f"Minutes: {(time.time() - start_time) / 60}")
