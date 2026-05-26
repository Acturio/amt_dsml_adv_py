"""Python helpers for the AMAT data science and machine learning lessons.

The original notes used R packages such as tidymodels, recipes, yardstick,
fairness, srvyr and lime.  This module keeps the translated R Markdown chunks
compact while still using standard Python libraries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import math

import numpy as np
import pandas as pd
from IPython.display import Image, display
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_recall_curve,
    r2_score,
    root_mean_squared_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def display_image(path: str | Path, width: int | None = None, height: int | None = None) -> None:
    """Display a local image from a Python R Markdown chunk.

    Static images in the book are now inserted with ``knitr::include_graphics``.
    This fallback uses matplotlib because knitr captures figures more reliably
    than raw ``IPython.display.Image`` objects in reticulate chunks.
    """

    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        image = mpimg.imread(str(path))
        image_height, image_width = image.shape[:2]
        fig_width = width / 96 if width else min(8, max(3, image_width / 150))
        fig_height = height / 96 if height else fig_width * image_height / image_width
        _, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(image)
        ax.axis("off")
        plt.tight_layout(pad=0)
    except Exception:
        display(Image(filename=str(path), width=width, height=height))


def show_missing(df: pd.DataFrame, title: str | None = None) -> pd.DataFrame:
    """Return a compact missing-value summary similar to DataExplorer::plot_missing."""

    summary = (
        df.isna()
        .mean()
        .rename("missing_rate")
        .mul(100)
        .reset_index()
        .rename(columns={"index": "variable"})
        .query("missing_rate > 0")
        .sort_values("missing_rate", ascending=False)
    )
    if title:
        print(title)
    if summary.empty:
        print("No hay valores perdidos.")
    else:
        display(summary)
    return summary


def load_telco(path: str | Path = "data/Churn.csv") -> pd.DataFrame:
    """Load the Telco churn data and coerce numeric columns used in the notes."""

    telco = pd.read_csv(path)
    if "TotalCharges" in telco.columns:
        telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors="coerce")
    return telco


def _rename_ames_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "SalePrice": "Sale_Price",
        "GrLivArea": "Gr_Liv_Area",
        "FullBath": "Full_Bath",
        "HalfBath": "Half_Bath",
        "BedroomAbvGr": "Bedroom_AbvGr",
        "KitchenAbvGr": "Kitchen_AbvGr",
        "TotRmsAbvGrd": "TotRms_AbvGrd",
        "YearBuilt": "Year_Built",
        "YearRemodAdd": "Year_Remod_Add",
        "YrSold": "Year_Sold",
        "TotalBsmtSF": "Total_Bsmt_SF",
        "FirstFlrSF": "First_Flr_SF",
        "SecondFlrSF": "Second_Flr_SF",
        "ThreeSsnPorch": "Three_season_porch",
        "OpenPorchSF": "Open_Porch_SF",
        "EnclosedPorch": "Enclosed_Porch",
        "PoolArea": "Pool_Area",
        "PoolQC": "Pool_QC",
        "ExterCond": "Exter_Cond",
        "BsmtFullBath": "Bsmt_Full_Bath",
        "BsmtHalfBath": "Bsmt_Half_Bath",
        "BsmtFinType1": "BsmtFin_Type_1",
        "BsmtFinType2": "BsmtFin_Type_2",
        "BsmtFinSF1": "BsmtFin_SF_1",
        "BsmtCond": "Bsmt_Cond",
        "BsmtExposure": "Bsmt_Exposure",
        "MasVnrType": "Mas_Vnr_Type",
        "MasVnrArea": "Mas_Vnr_Area",
        "LotFrontage": "Lot_Frontage",
        "LotArea": "Lot_Area",
        "HeatingQC": "Heating_QC",
        "SaleType": "Sale_Type",
        "MSZoning": "MS_Zoning",
        "BldgType": "Bldg_Type",
        "LandSlope": "Land_Slope",
        "LandContour": "Land_Contour",
        "LotShape": "Lot_Shape",
        "Condition1": "Condition_1",
        "MiscFeature": "Misc_Feature",
        "GarageType": "Garage_Type",
        "GarageFinish": "Garage_Finish",
        "GarageCond": "Garage_Cond",
    }
    return df.rename(columns=rename_map)


def _synthetic_ames(n: int = 2930, random_state: int = 4595) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame(
        {
            "Gr_Liv_Area": rng.normal(1500, 450, n).clip(450, 4500),
            "Full_Bath": rng.integers(1, 4, n),
            "Half_Bath": rng.integers(0, 2, n),
            "Bedroom_AbvGr": rng.integers(1, 6, n),
            "Kitchen_AbvGr": rng.integers(1, 3, n),
            "TotRms_AbvGrd": rng.integers(4, 12, n),
            "Year_Built": rng.integers(1900, 2011, n),
            "Year_Remod_Add": rng.integers(1950, 2011, n),
            "Year_Sold": rng.integers(2006, 2011, n),
            "Total_Bsmt_SF": rng.normal(1050, 350, n).clip(0, 3000),
            "First_Flr_SF": rng.normal(1150, 350, n).clip(350, 3000),
            "Second_Flr_SF": rng.choice([0, 0, 0, 500, 750, 1000], n),
            "Three_season_porch": rng.choice([0, 0, 0, 120], n),
            "Open_Porch_SF": rng.normal(45, 35, n).clip(0, 250),
            "Enclosed_Porch": rng.normal(20, 40, n).clip(0, 250),
            "Pool_Area": rng.choice([0, 0, 0, 0, 450], n),
            "Bsmt_Full_Bath": rng.integers(0, 2, n),
            "Bsmt_Half_Bath": rng.integers(0, 2, n),
            "BsmtFin_SF_1": rng.normal(450, 250, n).clip(0, 1800),
            "Mas_Vnr_Area": rng.normal(100, 90, n).clip(0, 600),
            "Lot_Frontage": rng.normal(70, 25, n).clip(20, 200),
            "Lot_Area": rng.normal(10000, 3500, n).clip(1500, 40000),
        }
    )
    categories = {
        "Alley": ["Grvl", "Pave", np.nan],
        "Pool_QC": ["Good", "Typical", np.nan],
        "Misc_Feature": ["Shed", "Gar2", np.nan],
        "Fence": ["Good", "Minimum", np.nan],
        "Garage_Finish": ["Fin", "RFn", "Unf", np.nan],
        "Garage_Cond": ["Poor", "Typical", "Good", np.nan],
        "Garage_Type": ["Attchd", "Detchd", "BuiltIn", np.nan],
        "Bsmt_Exposure": ["No", "Mn", "Av", "Gd", np.nan],
        "Bsmt_Cond": ["Poor", "Typical", "Good", "Excellent", np.nan],
        "BsmtFin_Type_1": ["Unf", "Rec", "BLQ", "ALQ", "GLQ", np.nan],
        "BsmtFin_Type_2": ["Unf", "Rec", "BLQ", "LwQ", np.nan],
        "Mas_Vnr_Type": ["None", "BrkFace", "Stone", np.nan],
        "Electrical": ["SBrkr", "FuseA", np.nan],
        "Heating_QC": ["Poor", "Fair", "Typical", "Good", "Excellent"],
        "Exter_Cond": ["Poor", "Fair", "Typical", "Good", "Excellent"],
        "Sale_Type": ["WD", "New", "COD", "Oth", "VWD"],
        "Neighborhood": [
            "North_Ames",
            "College_Creek",
            "Northridge",
            "Gilbert",
            "Somerset",
            "Old_Town",
        ],
        "Condition_1": ["Norm", "Feedr", "Artery", "PosN", "RRAn"],
        "Land_Slope": ["Gtl", "Mod", "Sev"],
        "Land_Contour": ["Lvl", "Bnk", "Low", "HLS"],
        "Lot_Shape": ["Regular", "Slightly_Irregular", "Moderately_Irregular", "Irregular"],
        "Heating": ["GasA", "GasW", "Grav", "Wall"],
        "MS_Zoning": ["Residential_Low_Density", "Residential_Medium_Density", "I_all"],
        "Bldg_Type": ["OneFam", "Duplex", "Twnhs"],
        "Foundation": ["PConc", "CBlock", "Wood", "Stone"],
        "Functional": ["Typ", "Min1", "Min2", "Maj1", "Maj2", "Mod"],
    }
    for col, values in categories.items():
        df[col] = rng.choice(values, n)
    price = (
        50000
        + 85 * df["Gr_Liv_Area"]
        + 35 * df["Total_Bsmt_SF"]
        + 900 * (df["Year_Built"] - 1900)
        + rng.normal(0, 25000, n)
    )
    df["Sale_Price"] = price.clip(35000, 800000)
    return df


def load_ames() -> pd.DataFrame:
    """Load Ames housing data from OpenML; fall back to a synthetic teaching set."""

    try:
        ames_raw = fetch_openml(name="house_prices", as_frame=True, parser="auto")
        frame = ames_raw.frame.copy()
        frame = _rename_ames_columns(frame)
        if "Sale_Price" not in frame.columns and "SalePrice" in frame.columns:
            frame = frame.rename(columns={"SalePrice": "Sale_Price"})
        for col in frame.select_dtypes(include="object").columns:
            frame[col] = frame[col].astype("category")
        return frame
    except Exception:
        return _synthetic_ames()


def load_biomass(n: int = 536, random_state: int = 19735) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame(
        {
            "dataset": np.where(np.arange(n) < int(n * 0.7), "Training", "Testing"),
            "carbon": rng.normal(48, 10, n),
            "hydrogen": rng.normal(6, 1.2, n),
            "oxygen": rng.normal(38, 9, n),
            "nitrogen": rng.normal(1.5, 0.8, n).clip(0, None),
            "sulfur": rng.normal(0.2, 0.15, n).clip(0, None),
        }
    )
    df["HHV"] = 0.35 * df["carbon"] + 1.2 * df["hydrogen"] - 0.15 * df["oxygen"] + rng.normal(0, 1.2, n)
    return df


def load_compas(n: int = 4_000, random_state: int = 2022) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    ethnicity = rng.choice(["Caucasian", "African-American", "Hispanic", "Asian"], n, p=[0.38, 0.42, 0.16, 0.04])
    base = pd.Series(ethnicity).map(
        {"Caucasian": -0.15, "African-American": 0.25, "Hispanic": 0.05, "Asian": -0.25}
    ).to_numpy()
    score = base + rng.normal(0, 0.85, n)
    probability = 1 / (1 + np.exp(-score))
    recid = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "ethnicity": ethnicity,
            "probability": probability,
            "Two_yr_Recidivism": np.where(recid == 1, "yes", "no"),
            "Two_yr_Recidivism_01": recid,
        }
    )


def _synthetic_margin_index(random_state: int = 2010) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    states = [
        "Aguascalientes",
        "Baja California",
        "Campeche",
        "Chiapas",
        "Chihuahua",
        "Ciudad de Mexico",
        "Guanajuato",
        "Guerrero",
        "Jalisco",
        "Mexico",
        "Nuevo Leon",
        "Oaxaca",
        "Puebla",
        "Queretaro",
        "Veracruz",
        "Yucatan",
    ]
    n = len(states)
    latent = np.linspace(-2, 2, n) + rng.normal(0, 0.35, n)
    df = pd.DataFrame({"NOM_ENT": states})
    for col, scale in {
        "ANALF": 3.0,
        "SPRIM": 5.0,
        "OVSDE": 2.0,
        "OVSEE": 1.8,
        "OVSAE": 2.5,
        "VHAC": 4.0,
        "OVPT": 3.2,
        "PL_5000": 5.0,
        "PO2SM": 4.5,
    }.items():
        df[col] = (50 + latent * scale + rng.normal(0, 1.2, n)).clip(0, 100)
    df["IM"] = df[["ANALF", "SPRIM", "OVSDE", "OVSEE", "OVSAE", "VHAC", "OVPT", "PL_5000", "PO2SM"]].mean(axis=1)
    df["GM"] = pd.cut(
        df["IM"],
        bins=[-np.inf, 45, 50, 55, 60, np.inf],
        labels=["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"],
    )
    df["LUGAR"] = np.arange(1, n + 1)
    df["AÑO"] = 2010
    df["POB_TOT"] = rng.integers(100_000, 8_000_000, n)
    return df


def load_margin_index(path: str | Path = "data/IMEF_2010.dbf") -> pd.DataFrame:
    """Read the CONAPO margin-index DBF used for the PCA examples."""

    try:
        import pyreadstat

        df, _ = pyreadstat.read_dbf(str(path))
        return df
    except Exception:
        try:
            from dbfread import DBF

            return pd.DataFrame(iter(DBF(str(path), load=True, encoding="latin1")))
        except Exception:
            return _synthetic_margin_index()


def collapse_values(series: pd.Series, mapping: dict[str, Iterable[str]]) -> pd.Series:
    out = series.astype("object").copy()
    for new_value, old_values in mapping.items():
        out.loc[out.isin(list(old_values))] = new_value
    return out.astype("category")


def preprocess_telco_frame(df: pd.DataFrame, keep_target: bool = True) -> pd.DataFrame:
    """Feature engineering equivalent to the Telco recipes used in the R notes."""

    out = df.copy()
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    service_cols = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in service_cols:
        if col in out.columns:
            out[col] = out[col].replace({"No phone service": "No", "No internet service": "No"})
    if "tenure" in out.columns:
        out["tenure_band"] = pd.cut(
            out["tenure"],
            bins=[0, 12, 24, 36, 48, 60, 72],
            labels=["0-1 year", "1-2 years", "2-3 years", "3-4 years", "4-5 years", "5-6 years"],
            include_lowest=True,
        )
        out = out.drop(columns=["tenure"])
    if "customerID" in out.columns:
        out = out.drop(columns=["customerID"])
    if not keep_target and "Churn" in out.columns:
        out = out.drop(columns=["Churn"])
    return out


def preprocess_ames_frame(df: pd.DataFrame, keep_target: bool = True, large: bool = False) -> pd.DataFrame:
    """Feature engineering equivalent to the main Ames recipes used in the notes."""

    out = _rename_ames_columns(df.copy())
    unknown_cols = [
        "Alley",
        "Pool_QC",
        "Misc_Feature",
        "Fence",
        "Garage_Finish",
        "Garage_Cond",
        "Garage_Type",
        "Bsmt_Exposure",
        "Bsmt_Cond",
        "BsmtFin_Type_1",
        "BsmtFin_Type_2",
        "Mas_Vnr_Type",
        "Electrical",
        "Heating_QC",
    ]
    for col in unknown_cols:
        if col in out.columns:
            out[col] = out[col].astype("object").fillna("Unknown")

    if "Year_Remod_Add" in out.columns:
        out["Year_Remod"] = out["Year_Remod_Add"]
    if "Three_season_porch" in out.columns:
        out["ThirdSsn_Porch"] = out["Three_season_porch"]
    if {"Bedroom_AbvGr", "Gr_Liv_Area"}.issubset(out.columns):
        out["Bedroom_AbvGr_ratio"] = out["Bedroom_AbvGr"] / out["Gr_Liv_Area"].replace(0, np.nan)
    if {"Year_Sold", "Year_Remod"}.issubset(out.columns):
        out["Age_House"] = out["Year_Sold"] - out["Year_Remod"]
    if {"Gr_Liv_Area", "Total_Bsmt_SF"}.issubset(out.columns):
        out["TotalSF"] = out["Gr_Liv_Area"] + out["Total_Bsmt_SF"]
    if {"Gr_Liv_Area", "TotRms_AbvGrd"}.issubset(out.columns):
        out["AvgRoomSF"] = out["Gr_Liv_Area"] / out["TotRms_AbvGrd"].replace(0, np.nan)
    if "Pool_Area" in out.columns:
        out["Pool"] = (out["Pool_Area"] > 0).astype(int)
    if "Exter_Cond" in out.columns:
        out["Exter_Cond"] = out["Exter_Cond"].replace({"Typical": "Good", "Excellent": "Good"})

    if large:
        if {"Full_Bath", "Bsmt_Full_Bath", "Half_Bath", "Bsmt_Half_Bath"}.issubset(out.columns):
            out["TotalBaths"] = out["Full_Bath"] + out["Bsmt_Full_Bath"] + 0.5 * (
                out["Half_Bath"] + out["Bsmt_Half_Bath"]
            )
        if {"TotalSF", "Lot_Area"}.issubset(out.columns):
            out["Porc_H_over_TotalSF"] = (out["TotalSF"] / out["Lot_Area"].replace(0, np.nan)) * 100
        if {"Enclosed_Porch", "ThirdSsn_Porch", "Open_Porch_SF"}.issubset(out.columns):
            out["Porch_SF"] = out["Enclosed_Porch"] + out["ThirdSsn_Porch"] + out["Open_Porch_SF"]
            out["Porch"] = out["Porch_SF"].gt(0).astype("category")

    drop_cols = [
        "First_Flr_SF",
        "Second_Flr_SF",
        "Year_Remod",
        "Year_Remod_Add",
        "Bsmt_Full_Bath",
        "Bsmt_Half_Bath",
        "Kitchen_AbvGr",
        "BsmtFin_Type_1_Unf",
        "Total_Bsmt_SF",
        "Pool_Area",
        "Gr_Liv_Area",
        "Sale_Type_Oth",
        "Sale_Type_VWD",
        "Porch_SF",
    ]
    out = out.drop(columns=[col for col in drop_cols if col in out.columns], errors="ignore")
    if not keep_target and "Sale_Price" in out.columns:
        out = out.drop(columns=["Sale_Price"])
    return out


class DataFrameFunctionTransformer(BaseEstimator, TransformerMixin):
    """Apply a pandas feature-engineering function inside a scikit-learn Pipeline."""

    def __init__(self, func, kw_args: dict | None = None):
        self.func = func
        self.kw_args = kw_args or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(pd.DataFrame(X).copy(), **self.kw_args)


def split_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target]), df[target]


def tabular_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), make_column_selector(dtype_include=np.number)),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_ames_pipeline(model, scale_numeric: bool = True, large: bool = False) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", DataFrameFunctionTransformer(preprocess_ames_frame, {"keep_target": False, "large": large})),
            ("preprocess", tabular_preprocessor(scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def make_telco_pipeline(model, scale_numeric: bool = True) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", DataFrameFunctionTransformer(preprocess_telco_frame, {"keep_target": False})),
            ("preprocess", tabular_preprocessor(scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def make_workflow_set(recipes: dict[str, dict], models: dict[str, object]) -> dict[str, Pipeline]:
    """Create a sklearn workflow set from recipe specs and model specs."""

    workflows = {}
    for recipe_name, recipe_spec in recipes.items():
        recipe_args = {key: value for key, value in recipe_spec.items() if key != "name"}
        for model_name, model in models.items():
            workflows[f"{recipe_name}_{model_name}"] = make_ames_pipeline(clone(model), **recipe_args)
    return workflows


def model_key_from_workflow_id(workflow_id: str, param_spaces: dict[str, dict]) -> str:
    """Infer the model key for a workflow id such as ``receta_knn_svm_rbf``."""

    return next(key for key in sorted(param_spaces, key=len, reverse=True) if workflow_id.endswith(key))


def tune_workflow_set(
    workflows: dict[str, Pipeline],
    param_spaces: dict[str, dict],
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scoring,
    refit=True,
    n_iter: int = 10,
    random_state: int = 20220603,
    n_jobs: int = -1,
) -> dict[str, RandomizedSearchCV]:
    """Tune many sklearn workflows with their corresponding parameter spaces."""

    searches = {}
    for workflow_id, pipeline in workflows.items():
        model_key = model_key_from_workflow_id(workflow_id, param_spaces)
        search = RandomizedSearchCV(
            estimator=clone(pipeline),
            param_distributions=param_spaces[model_key],
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            refit=refit,
            random_state=random_state,
            n_jobs=n_jobs,
            return_train_score=False,
        )
        search.fit(X, y)
        searches[workflow_id] = search
    return searches


def collect_search_results(searches: dict[str, RandomizedSearchCV], sort_by: str = "rsq") -> pd.DataFrame:
    """Collect best parameters and best CV metrics from a workflow-set search."""

    rows = []
    for workflow_id, search in searches.items():
        cv_results = pd.DataFrame(search.cv_results_)
        best = cv_results.loc[search.best_index_]
        row = {"wflow_id": workflow_id}
        for col in cv_results.columns:
            if col.startswith("mean_test_"):
                metric = col.removeprefix("mean_test_")
                value = best[col]
                if metric in {"rmse", "mae", "mape"}:
                    value = -value
                row[metric] = value
        if "mean_test_score" in cv_results:
            row["score"] = best["mean_test_score"]
        row.update(search.best_params_)
        rows.append(row)

    out = pd.DataFrame(rows)
    ascending = sort_by in {"rmse", "mae", "mape"}
    if sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=ascending)
    return out.reset_index(drop=True)


def top_estimators_for_stacking(
    searches: dict[str, RandomizedSearchCV],
    results: pd.DataFrame,
    n: int = 4,
    score_col: str = "rsq",
) -> list[tuple[str, Pipeline]]:
    """Return the top tuned estimators as named sklearn estimators for stacking."""

    ascending = score_col in {"rmse", "mae", "mape"}
    top_ids = results.sort_values(score_col, ascending=ascending).head(n)["wflow_id"]
    return [(workflow_id, clone(searches[workflow_id].best_estimator_)) for workflow_id in top_ids]


def stacking_weights(stack: StackingRegressor) -> pd.DataFrame:
    """Return a tidy table with meta-model coefficients from a fitted stack."""

    coefs = np.ravel(stack.final_estimator_.coef_)
    return pd.DataFrame({"model": [name for name, _ in stack.estimators], "weight": coefs})


def regression_metrics(y_true, y_pred) -> pd.DataFrame:
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cor = np.corrcoef(y_true, y_pred)[0, 1]
    ccc = (2 * cor * y_true.std() * y_pred.std()) / (
        y_true.var() + y_pred.var() + (y_true.mean() - y_pred.mean()) ** 2
    )
    return pd.DataFrame(
        {
            "metric": ["rmse", "mae", "mape", "rsq", "ccc"],
            "estimate": [rmse, mae, mape, r2, ccc],
        }
    )


def classification_metrics(y_true, prob_positive, positive_label="Yes") -> pd.DataFrame:
    y = pd.Series(y_true).eq(positive_label).astype(int)
    return pd.DataFrame(
        {
            "metric": ["roc_auc", "pr_auc"],
            "estimate": [roc_auc_score(y, prob_positive), average_precision_score(y, prob_positive)],
        }
    )


def roc_curve_table(y_true, prob_positive, positive_label="Yes") -> pd.DataFrame:
    y = pd.Series(y_true).eq(positive_label).astype(int)
    fpr, tpr, thresholds = roc_curve(y, prob_positive)
    return pd.DataFrame({"threshold": thresholds, "specificity": 1 - fpr, "sensitivity": tpr})


def pr_curve_table(y_true, prob_positive, positive_label="Yes") -> pd.DataFrame:
    y = pd.Series(y_true).eq(positive_label).astype(int)
    precision, recall, thresholds = precision_recall_curve(y, prob_positive)
    thresholds = np.append(thresholds, np.nan)
    return pd.DataFrame({"threshold": thresholds, "precision": precision, "recall": recall})


def best_cv_results(search, metric: str | None = None, n: int = 10) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_)
    rank_col = f"rank_test_{metric}" if metric else "rank_test_score"
    if rank_col not in results:
        rank_col = "rank_test_score"
    cols = [col for col in results.columns if col.startswith("param_") or col.startswith("mean_test") or col.startswith("std_test")]
    return results.sort_values(rank_col).loc[:, cols].head(n)


def permutation_importance_table(model, X, y, scoring: str, n_repeats: int = 10, random_state: int = 123) -> pd.DataFrame:
    result = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state)
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = [f"x{i}" for i in range(result.importances_mean.size)]
    return (
        pd.DataFrame({"Variable": names, "Importance": result.importances_mean, "StDev": result.importances_std})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def sample_size_proportion(
    e: float = 0.04,
    p: float = 0.5,
    alpha: float = 0.95,
    N: int = 100_000,
    deff: float = 1.5,
    tnr: float = 0.10,
) -> int:
    z = stats.norm.ppf(1 - (1 - alpha) / 2)
    m = p * (1 - p) * (z / e) ** 2
    n = (m / (1 + m / N)) * deff / (1 - tnr)
    return math.ceil(n)


def allocate_proportional(frame: pd.DataFrame, strata: str, n: int) -> pd.DataFrame:
    distribution = frame.groupby(strata).size().rename("N_h").reset_index()
    distribution["prop"] = distribution["N_h"] / distribution["N_h"].sum()
    distribution["n_h"] = np.maximum(1, np.round(distribution["prop"] * n).astype(int))
    return distribution


def stratified_srswor(frame: pd.DataFrame, strata: str | None, size, random_state: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    data = frame.copy().reset_index(drop=True)
    data["ID_unit"] = np.arange(1, len(data) + 1)
    if strata is None:
        n = int(size)
        selected = data.sample(n=n, replace=False, random_state=random_state)
        selected["Prob"] = n / len(data)
        return selected.assign(factor=lambda x: 1 / x["Prob"])

    if isinstance(size, pd.DataFrame):
        alloc = size.set_index(strata)["n_h"].to_dict()
    elif isinstance(size, dict):
        alloc = size
    else:
        levels = data[strata].drop_duplicates().to_list()
        alloc = dict(zip(levels, size))

    samples = []
    for level, group in data.groupby(strata):
        n_h = min(int(alloc.get(level, 0)), len(group))
        if n_h <= 0:
            continue
        chosen = group.iloc[rng.choice(len(group), size=n_h, replace=False)].copy()
        chosen["Prob"] = n_h / len(group)
        samples.append(chosen)
    return pd.concat(samples, ignore_index=True).assign(factor=lambda x: 1 / x["Prob"])


def weighted_proportions(
    frame: pd.DataFrame,
    value: str,
    weight: str,
    group: list[str] | str | None = None,
    alpha: float = 0.95,
) -> pd.DataFrame:
    group_cols = [] if group is None else ([group] if isinstance(group, str) else group)
    z = stats.norm.ppf(1 - (1 - alpha) / 2)
    rows = []
    grouped = frame.groupby(group_cols + [value], dropna=False) if group_cols else frame.groupby(value, dropna=False)
    totals = frame.groupby(group_cols)[weight].sum() if group_cols else pd.Series({"__all__": frame[weight].sum()})
    for keys, part in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        strata_key = keys[:-1] if group_cols else ("__all__",)
        denom = totals.loc[strata_key if len(strata_key) > 1 else strata_key[0]]
        prop = part[weight].sum() / denom
        n_eff = denom**2 / (frame.loc[part.index, weight] ** 2).sum()
        se = math.sqrt(max(prop * (1 - prop), 0) / max(n_eff, 1))
        row = dict(zip(group_cols, keys[:-1]))
        row[value] = keys[-1]
        row.update({"prop": prop, "prop_low": max(0, prop - z * se), "prop_upp": min(1, prop + z * se), "prop_cv": se / prop if prop else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def _binary_arrays(data, outcome: str, probs: str, cutoff: float, positive=1):
    y_true = data[outcome].eq(positive).astype(int) if not np.issubdtype(data[outcome].dtype, np.number) else data[outcome].astype(int)
    y_pred = (data[probs] >= cutoff).astype(int)
    return y_true, y_pred


def fairness_table(
    data: pd.DataFrame,
    outcome: str,
    group: str,
    probs: str,
    cutoff: float = 0.5,
    base: str | None = None,
    positive=1,
) -> pd.DataFrame:
    y_true, y_pred = _binary_arrays(data, outcome, probs, cutoff, positive=positive)
    rows = []
    for level, idx in data.groupby(group).groups.items():
        yt = y_true.loc[idx]
        yp = y_pred.loc[idx]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        group_size = tp + fp + tn + fn
        rows.append(
            {
                group: level,
                "group_size": group_size,
                "demographic_parity": (tp + fp) / group_size if group_size else np.nan,
                "proportional_parity": (tp + fp) / group_size if group_size else np.nan,
                "equal_odds": tp / (tp + fn) if (tp + fn) else np.nan,
                "predictive_rate_parity": tp / (tp + fp) if (tp + fp) else np.nan,
                "accuracy_parity": (tp + tn) / group_size if group_size else np.nan,
                "fnr_parity": fn / (tp + fn) if (tp + fn) else np.nan,
                "fpr_parity": fp / (tn + fp) if (tn + fp) else np.nan,
                "npv_parity": tn / (tn + fn) if (tn + fn) else np.nan,
                "specificity_parity": tn / (tn + fp) if (tn + fp) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    base = base or out[group].iloc[0]
    metric_cols = [col for col in out.columns if col.endswith("parity") or col == "equal_odds"]
    base_values = out.loc[out[group].eq(base), metric_cols].iloc[0]
    for col in metric_cols:
        out[f"{col}_ratio"] = out[col] / base_values[col]
    return out


def metric_view(table: pd.DataFrame, metric: str, group: str = "ethnicity") -> pd.DataFrame:
    cols = [group, metric, f"{metric}_ratio"]
    return table.loc[:, [col for col in cols if col in table.columns]]
