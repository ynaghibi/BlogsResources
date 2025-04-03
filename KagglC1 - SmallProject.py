###

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
from IPython.display import display
from pathlib import Path
sLocal_Folder_Path = Path(__file__).parent.resolve()
train_file_path = sLocal_Folder_Path / "train.csv" # Uses OS-appropriate separator
predict_file_path = sLocal_Folder_Path / "test.csv"
housing = pd.read_csv(train_file_path)
housing_unknown = pd.read_csv(predict_file_path)
display(housing)
display(housing_unknown)

#%%
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(
    housing, test_size = 0.05, random_state = 42
)
housing = train_set.drop("SalePrice", axis=1) #copies content except "SalePrice"-column
housing_targets = train_set["SalePrice"].copy() #copies only "SalePrice"-column

housing_final_test = test_set.drop("SalePrice", axis=1)
housing_targets_final_test = test_set["SalePrice"].copy()

# %%
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import ColumnTransformer

#%% Test
def increment_func(x):
    return x + 1
def double_func(x):
    return x * 2
example_pipe = make_pipeline(
    FunctionTransformer(func = increment_func),
    FunctionTransformer(func = double_func)
)
example_df = housing[["OverallQual", "GarageArea"]]
example_pipe.fit(example_df)
example_trafo = example_pipe.transform(example_df)
display(example_df)
display(example_trafo)

#%%
target_trafo = make_pipeline(
    SimpleImputer(strategy = "mean", add_indicator=True),
    FunctionTransformer(func = np.log, inverse_func = np.exp),
    StandardScaler()
)
feature_log_pipeline = make_pipeline(
    SimpleImputer(strategy="mean"),
    FunctionTransformer(func = np.log),
    StandardScaler()
)
feature_pipeline = make_pipeline(
    SimpleImputer(strategy="mean"),
    StandardScaler()
)

#%%
def column_ratio(X):
    return X[:,[0]] / X[:,[1]]
feature_ratio = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(func = column_ratio),
    StandardScaler()
)

#%%
column_trafo = ColumnTransformer(
    transformers = [
        ('num_log', feature_log_pipeline, ["GrLivArea"]),
        ('ratio', feature_ratio, ["BedroomAbvGr", "GrLivArea"]),
        ('num', feature_pipeline, ["OverallQual", "GarageArea"])
    ],
    remainder = "drop"
)

#%%
regressor_pipeline = Pipeline([
    ('features', column_trafo),
    ('linear', LinearRegression())
])

#%%
model = TransformedTargetRegressor(
    transformer = target_trafo,
    regressor = regressor_pipeline
)
model.fit(housing, housing_targets)
housing_predicted_prices = model.predict(housing_final_test)

#%%
from sklearn.metrics import root_mean_squared_error
lin_rmse = root_mean_squared_error(
    housing_targets_final_test, housing_predicted_prices
)
print((lin_rmse / 1e3).round(2), "thousand dollar RMSE for housing prices")

# %% getting submission results for Kaggle competition
housing_predicted_prices = model.predict(housing_unknown)
submission = pd.DataFrame({
    'Id': housing_unknown['Id'],
    'SalePrice': housing_predicted_prices
})
submission.to_csv(sLocal_Folder_Path / 'submission.csv', index=False)

# %%
