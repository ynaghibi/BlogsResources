###

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
from pathlib import Path
from IPython.display import display
sLocal_Folder_Path = "C:/.../" #insert your folder name here
housing = pd.read_csv(Path( sLocal_Folder_Path + "train.csv"))
housing_test = pd.read_csv(Path( sLocal_Folder_Path + "test.csv"))
display(housing)
display(housing_test)

# %%
housing.hist(bins=50, figsize=(30,25))
plt.show()

# %%
heavy_tailed_features = ["LotFrontage", "LotArea", "1stFlrSF", "TotalBsmtSF", "GrLivArea"]
housing[heavy_tailed_features].hist(bins=50, figsize=(12,8))
plt.show()

# %% encode categories to numbers (fibonacci-numbers are suitable for human scoring)
from KagglDataC1 import *
ranked_category_columns = ["BsmtQual", "BsmtCond", "BsmtExposure", 
    "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual", 
    "Functional", "FireplaceQu", "GarageFinish", "GarageQual",
    "GarageCond", "PavedDrive", "PoolQC", "Fence", "ExterCond", "ExterQual"
]

def transform_categories_to_ranked(data):
    for col in ranked_category_columns:
        data[f"Ranked_{col}"] = data[col].map(globals()[f"fibonacci_mapping_{col}"])
    data = data.drop(columns=ranked_category_columns)
    return data

housing = transform_categories_to_ranked(housing)
housing_test = transform_categories_to_ranked(housing_test)

# %%
display(housing_test.info())
display(housing.info())

# %%
# amount of columns may become too large to be viewed as scrollable elements
# therefore we need to set various settings with pd.set_option(...) 
pd.set_option('display.max_info_columns', 250)
pd.set_option('display.max_rows', 250)

# %%
housing.describe()

# %%
#(housing["KitchenAbvGr"]).value_counts().sort_index()
#(housing["LotArea"]).describe()
(housing["OverallQual"]).value_counts().sort_index()

# %%
housing_original = housing.copy() #important to revert housing back to current state

# %%
corr_matrix = housing.corr(numeric_only = True)
corr_matrix["SalePrice"].sort_values(ascending = False)

# %% educated guess modifications to columns:
housing["bath_sum"] = \
    housing["FullBath"] + housing["HalfBath"] \
    + housing["BsmtFullBath"] + housing["BsmtHalfBath"]
housing["areaquality_product"] = housing["GrLivArea"] * housing["OverallQual"]
housing["garage_product"] = housing["Ranked_PavedDrive"] \
    * housing["Ranked_GarageFinish"] * housing["Ranked_GarageQual"] \
    * housing["GarageCars"] * housing["GarageArea"]
housing["bedrooms_ratio"] = housing["BedroomAbvGr"] / housing["GrLivArea"]
housing["roomquality_product"] = housing["Ranked_HeatingQC"] \
    * housing["TotRmsAbvGrd"]
housing["bath_kitchen_ratio"] = np.where(
    housing["KitchenAbvGr"] != 0,  # Condition
    (housing["bath_sum"]) / housing["KitchenAbvGr"],  # True: Perform division
    (housing["bath_sum"]) / housing["TotRmsAbvGrd"]
)
housing["bath_bedroom_ratio"] = np.where(
    housing["BedroomAbvGr"] != 0,  # Condition
    (housing["bath_sum"]) / housing["BedroomAbvGr"],  # True: Perform division
    (housing["bath_sum"]) / housing["TotRmsAbvGrd"]
)

#correlation is a good indicator for strata categories
corr_matrix = housing.corr(numeric_only = True)
corr_matrix["SalePrice"].sort_values(ascending = False)

# %%
housing.info() #yet another check of the dataset

# %%
from pandas.plotting import scatter_matrix
attributes = ["SalePrice", "garage_product", "areaquality_product", 
    "roomquality_product", "bedrooms_ratio",
]
scatter_matrix(housing[attributes], figsize=(10, 10))
plt.show()

# %% testing...
display(housing["areaquality_product"].describe())
housing["bedrooms_ratio"].describe()

# %% use strongly correlated features for strata:
strata_cat_1 = pd.cut(
    housing["areaquality_product"],
    bins = [0, 5e3, 8e3, 12e3, np.inf],
    labels = [1,2,3,4],
    include_lowest = True #prevents possible NaN generation (e.g. if value is exactly 0, it will be Nan, because the edges (like 0 in this case) are excluded)
)
strata_cat_2 = pd.cut(
    housing["bedrooms_ratio"],
    bins = [0.0, 16e-4, 21e-4, np.inf],
    labels = [1,2,3],
    include_lowest = True
)
#strata_cat_2 = strata_cat_1 #use this only to disable influence of strata_cat_2 (otherwise outcomment this line)
strata_cat_1.value_counts().sort_index().plot.bar(grid = True)
plt.show()
strata_cat_2.value_counts().sort_index().plot.bar(grid = True)
plt.show()

# %%
has_nan = strata_cat_2.isna().any().any()  # Returns True if any NaN exists
print("Does the DataFrame contain NaN values?", has_nan)

# %%
sStrataCat = "Strata_Cat"
housing[sStrataCat] = strata_cat_1.astype(str) + "_" + strata_cat_2.astype(str)
housing[sStrataCat].value_counts().sort_index().plot.bar(grid = True)
#shows that some categories only have small amount of data (underrepresented strata)
#therefore we fix this by using the cell below

# %% use this cell in case some categories have don't have enough instances
iMinCounts = 100
housing_stratacat_counts = housing[sStrataCat].value_counts()
indices_of_small_housing_stratacat_counts = housing_stratacat_counts[
    housing_stratacat_counts < iMinCounts # Threshold: fewer than iMinCounts samples
].index

# Combine small categories into "Other"
housing[sStrataCat] = housing[sStrataCat].apply(
    lambda x: 'Other' if x in indices_of_small_housing_stratacat_counts else x
)
housing[sStrataCat].value_counts().sort_index().plot.bar(grid = True)
plt.show()

# %%
# Important step: get strata category and then revert housing back to clean data
from sklearn.model_selection import train_test_split
housing_strata_category = housing[sStrataCat].copy()
housing = housing_original
strat_train_set, strat_test_set = train_test_split(
    housing, test_size = 0.15, stratify = housing_strata_category, random_state = 42
)
housing = strat_train_set.drop("SalePrice", axis=1) #copies content except "SalePrice"-column
housing_labels = strat_train_set["SalePrice"].copy() #copies only "SalePrice"-column

# %% understand non-numerical categories better:
housing_cat_columns = housing.select_dtypes(include=[object])
for column in housing_cat_columns:
    print(f"Value counts for {column}:")
    print(housing_cat_columns[column].value_counts())
    print("-" * 30)

# %%
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat = housing.select_dtypes(include=['object', 'category'])
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat
#TODO: we do not need one hot encode like this for future code. it is just a demo

# %%
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import FunctionTransformer

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.utils.validation import check_array, check_is_fitted

list_trafo_columns = [
    "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath",
    "GrLivArea", "OverallQual",
    "Ranked_PavedDrive", "Ranked_GarageFinish", "Ranked_GarageQual", "GarageCars", "GarageArea",
    "BedroomAbvGr", "GrLivArea",
    "Ranked_HeatingQC", "GrLivArea",
    "KitchenAbvGr", "BedroomAbvGr", "TotRmsAbvGrd",
]
inverse_list_trafo_columns = {
    value: index for index, value in enumerate(list_trafo_columns)
}

class ColumnFormulaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
            sum = [], product = [], denominator = [], altdenominator = [],
            sumweights = []
        ):
        self.sum = sum
        self.product = product
        self.denominator = denominator
        self.altdenominator = altdenominator
        self.sumweights = sumweights
    def fit(self, X, y=None, sample_weight=None):
        self.n_features_in_ = X.shape[1]
        return self
    def transform(self, X):
        assert self.n_features_in_ == X.shape[1]
        #calculate nominator:
        numerator = np.zeros(X.shape[0])
        for id, col in enumerate(self.sum):
            if id < len(self.sumweights):
                weight = self.sumweights[id]
            else:
                weight = 1
            numerator += X[:,inverse_list_trafo_columns[col]] * weight
        if self.product:
            prodnumerator = np.ones(X.shape[0])
            for col in self.product:
                prodnumerator *= X[:,inverse_list_trafo_columns[col]]
            numerator += prodnumerator
        #calculate denominator:
        if self.denominator:
            denominator = X[:, inverse_list_trafo_columns[self.denominator[0]]]
            if self.altdenominator:
                altdenominator = \
                    X[:, inverse_list_trafo_columns[self.altdenominator[0]]]
                denominator[denominator == 0] = altdenominator[denominator == 0]
            result = numerator / denominator
        else:
            result = numerator
        return result.reshape(-1, 1) #convert result from 1D to 2D NumPy array
    def get_feature_names_out(self, names=None):
        return ["formula"]

def make_pipeline_with_formula(
        sum = [], product = [], 
        denominator = [], altdenominator = []
    ):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "ratio",
            ColumnFormulaTransformer(
                sum = sum, product = product, 
                denominator = denominator, altdenominator = altdenominator
            )
        ),
        ("scaler", StandardScaler())
    ])

# example code:
# ColumnFormulaTransformer(product = ["GrLivArea", "OverallQual"])

# Reminder:
# bath_sum = FullBath + HalfBath + BsmtFullBath + BsmtHalfBath
# areaquality_product = GrLivArea * OverallQual
# garage_product = Ranked_PavedDrive * Ranked_GarageFinish * Ranked_GarageQual
#       * GarageCars * GarageArea
# bedroom_ratio = BedroomAbvGr / GrLivArea #or bedroom_area_ratio
# roomquality_product = Ranked_HeatingQC * TotRmsAbvGrd
# bath_kitchen_ratio = bath_sum / KitchenAbvGr or TotRmsAbvGrd
# bath_bedroom_ratio = bath_sum / BedroomAbvGr or TotRmsAbvGrd

bathsum_list = ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
ColumnTransformer_TupleList = [
    ("bath", make_pipeline_with_formula(
            sum = bathsum_list
        ), list_trafo_columns
    ),
    ("areaquality", make_pipeline_with_formula(
            product = ["GrLivArea", "OverallQual"]
        ), list_trafo_columns
    ),
    ("garage", make_pipeline_with_formula(
            product = ["Ranked_PavedDrive", "Ranked_GarageFinish",
                        "Ranked_GarageQual", "GarageCars", "GarageArea"]
        ), list_trafo_columns
    ),
    ("bedroom", make_pipeline_with_formula(
            product = ["BedroomAbvGr"], denominator = ["GrLivArea"]
        ), list_trafo_columns
    ),
    ("roomquality", make_pipeline_with_formula(
            product = ["Ranked_HeatingQC", "TotRmsAbvGrd"]
        ), list_trafo_columns
    ),
    ("bath_kitchen", make_pipeline_with_formula(
            sum = bathsum_list,
            denominator = ["KitchenAbvGr"],
            altdenominator = ["TotRmsAbvGrd"]
        ), list_trafo_columns
    ),
    ("bath_bedroom", make_pipeline_with_formula(
            sum = bathsum_list,
            denominator = ["BedroomAbvGr"],
            altdenominator = ["TotRmsAbvGrd"]
        ), list_trafo_columns
    ),
]
#example code:
#preprocessing = ColumnTransformer(ColumnTransformer_TupleList)

inverse_list_trafo_columns

# %%
def safe_log(x):
    return np.log(np.where(x <= 0, 1e-10, x))
log_pipeline = make_pipeline(
    SimpleImputer(strategy = "median"),
    #FunctionTransformer(np.log, feature_names_out = "one-to-one"),
    FunctionTransformer(safe_log, feature_names_out = "one-to-one"),
    StandardScaler()
)
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)
default_num_pipeline = make_pipeline(
    SimpleImputer(strategy = "median"),
    StandardScaler()
)

ColumnTransformer_TupleList.extend([
    ("log", log_pipeline, heavy_tailed_features),
    ("cat", cat_pipeline, make_column_selector(dtype_include = object)),
])
preprocessing = ColumnTransformer(
    ColumnTransformer_TupleList, remainder = default_num_pipeline
)

# housing_prepared is for testing purposes only
housing_prepared = preprocessing.fit_transform(housing)
df_housing_prepared = pd.DataFrame(
    housing_prepared,
    columns = preprocessing.get_feature_names_out(), #columns=housing.columns,
    index = housing.index
)
df_housing_prepared

# %%
from sklearn.tree import DecisionTreeRegressor
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)
from sklearn.model_selection import cross_val_score
tree_rmses = -cross_val_score(
    tree_reg,
    housing,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

# %%
pd.Series(tree_rmses).describe()

# %%
from sklearn.ensemble import RandomForestRegressor
forest_reg = make_pipeline(
    preprocessing,
    RandomForestRegressor(random_state=42)
)
forest_rmses = -cross_val_score(
    forest_reg, housing, housing_labels,
    scoring="neg_root_mean_squared_error", cv=10
)

# %%
pd.Series(forest_rmses).describe()

# %%
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])
param_grid = [
    {
        'preprocessing__bath__ratio__sumweights': [
            #["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
            [1.1, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.9],
            [1.0, 1.0, 1.0, 1.0], #default
            [1.0, .5, .5, .25],
            [1.0, 1.0, .5, .5],
            [1.0, 1.0, .0, .0],
            [1.0, 1.0, 1.0, .5],
        ],
        # for preprocessing__bath_kitchen__ratio__sumweights the default is always optimal for some reason
        # for preprocessing__bath_bedroom__ratio__sumweights it is [1.0, 1.0, 0.5, 0.5]
        # for preprocessing__bath__ratio__sumweights it is [1.0, 1.0, 1.0, 0.5]
    },
]
grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=3, scoring='neg_root_mean_squared_error'
)
grid_search.fit(housing, housing_labels)

# %%
grid_search.best_params_

# %%
cv_rmse_scores = -grid_search.cv_results_['mean_test_score']
rmse_summary = pd.Series(cv_rmse_scores).describe()
rmse_summary

# %%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])

class CustomSumweightsSampler:
    def rvs(self, random_state=None):
        #["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
        return [1.0, uniform(0.0, 1.0).rvs(random_state=random_state), 
                uniform(0.0, 1.0).rvs(random_state=random_state), 
                uniform(0.0, 1.0).rvs(random_state=random_state)]

param_distribs = {
    'preprocessing__bath_kitchen__ratio__sumweights': CustomSumweightsSampler(),
    'preprocessing__bath_bedroom__ratio__sumweights': CustomSumweightsSampler(),
    'preprocessing__bath__ratio__sumweights': CustomSumweightsSampler(),
}

rnd_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions = param_distribs,
    n_iter=15,  # Number of parameter settings that are sampled
    cv=10, #cv = 3 or 5 is more usual, but a higher cv is recommended for small datasets in order to reduce the variance of the estimate
    scoring='neg_root_mean_squared_error',
    random_state=42
)
rnd_search.fit(housing, housing_labels)

# %%
rnd_search.best_params_

# %%
cv_rmse_scores = -rnd_search.cv_results_['mean_test_score']
rmse_summary = pd.Series(cv_rmse_scores).describe()
rmse_summary

# %%
from sklearn.metrics import root_mean_squared_error
housing_predicted_prices = rnd_search.predict(housing) #numpy.ndarray
housing_real_prices = housing_labels #pandas.core.series.Series
tree_rmse = root_mean_squared_error(housing_labels, housing_predicted_prices)
tree_rmse

# %% getting submission results for Kaggle competition
housing_predicted_prices = rnd_search.predict(housing_test)
submission = pd.DataFrame({
    'Id': housing_test['Id'],
    'SalePrice': housing_predicted_prices
})
submission.to_csv(sLocal_Folder_Path + '/submission.csv', index=False)

# %%
