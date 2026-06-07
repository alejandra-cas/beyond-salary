import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm


def _load_statsmodels():
    import statsmodels.api as sm

    return sm


class SimpleSummary:
    def __init__(self, text):
        self._text = text

    def as_text(self):
        return self._text

    def __str__(self):
        return self._text


class ConditionalLogitResults:
    def __init__(
        self,
        params,
        bse,
        pvalues,
        converged,
        nobs,
        llf,
        llnull,
        model_name,
        dropped_groups,
        used_groups,
    ):
        self.params = params
        self.bse = bse
        self.pvalues = pvalues
        self.converged = converged
        self.nobs = nobs
        self.llf = llf
        self.llnull = llnull
        self.prsquared = np.nan if llnull == 0 else 1 - (llf / llnull)
        self.model_name = model_name
        self.dropped_groups = dropped_groups
        self.used_groups = used_groups

    def summary(self):
        summary_df = pd.DataFrame(
            {
                "coef": self.params,
                "std err": self.bse,
                "p>|z|": self.pvalues,
            }
        )
        lines = [
            self.model_name,
            f"Converged: {self.converged}",
            f"Observations: {int(self.nobs)}",
            f"Groups used: {self.used_groups}",
            f"Groups dropped (no within-group outcome variation): {self.dropped_groups}",
            f"Log-Likelihood: {self.llf:.4f}",
            f"Pseudo R-squared: {self.prsquared:.4f}",
            "",
            summary_df.to_string(float_format=lambda x: f"{x:0.4f}"),
        ]
        return SimpleSummary("\n".join(lines))

    def conf_int(self, alpha=0.05):
        critical_value = norm.ppf(1 - alpha / 2)
        lower = self.params - critical_value * self.bse
        upper = self.params + critical_value * self.bse
        return pd.DataFrame({0: lower, 1: upper})


def _combine_log_n_choose_k(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def _group_log_denominator_and_inclusion_probs(linear_predictor, successes):
    group_size = len(linear_predictor)
    prefix = np.full((group_size + 1, successes + 1), -np.inf)
    suffix = np.full((group_size + 1, successes + 1), -np.inf)
    prefix[0, 0] = 0.0
    suffix[group_size, 0] = 0.0

    for idx in range(group_size):
        prefix[idx + 1] = prefix[idx]
        upper = min(successes, idx + 1)
        for degree in range(upper, 0, -1):
            prefix[idx + 1, degree] = np.logaddexp(
                prefix[idx + 1, degree],
                linear_predictor[idx] + prefix[idx, degree - 1],
            )

    for idx in range(group_size - 1, -1, -1):
        suffix[idx] = suffix[idx + 1]
        upper = min(successes, group_size - idx)
        for degree in range(upper, 0, -1):
            suffix[idx, degree] = np.logaddexp(
                suffix[idx, degree],
                linear_predictor[idx] + suffix[idx + 1, degree - 1],
            )

    log_denominator = prefix[group_size, successes]
    if not np.isfinite(log_denominator):
        raise ValueError("Conditional logit denominator is not finite.")

    inclusion_probs = np.zeros(group_size)
    for idx in range(group_size):
        log_partial_sum = -np.inf
        for degree in range(successes):
            log_partial_sum = np.logaddexp(
                log_partial_sum,
                prefix[idx, degree] + suffix[idx + 1, successes - 1 - degree],
            )
        inclusion_probs[idx] = np.exp(
            linear_predictor[idx] + log_partial_sum - log_denominator
        )

    inclusion_probs = np.clip(inclusion_probs, 0.0, 1.0)
    return log_denominator, inclusion_probs


def _prepare_conditional_groups(X, y, groups):
    grouped = []
    dropped_groups = 0

    for _, group_idx in pd.Series(np.arange(len(groups))).groupby(groups).groups.items():
        idx = np.asarray(list(group_idx), dtype=int)
        y_group = y[idx]
        successes = int(y_group.sum())
        if len(idx) < 2 or successes == 0 or successes == len(idx):
            dropped_groups += 1
            continue

        grouped.append(
            {
                "X": X[idx],
                "y": y_group,
                "successes": successes,
                "size": len(idx),
            }
        )

    if not grouped:
        raise ValueError("No groups with within-group outcome variation remain for conditional logit.")

    nobs = int(sum(group["size"] for group in grouped))
    llnull = -sum(_combine_log_n_choose_k(group["size"], group["successes"]) for group in grouped)
    return grouped, dropped_groups, nobs, float(llnull)


def _fit_conditional_logit(X, y, groups, column_names, model_name):
    grouped, dropped_groups, nobs, llnull = _prepare_conditional_groups(X, y, groups)

    def objective(beta):
        neg_loglike = 0.0
        gradient = np.zeros_like(beta)

        for group in grouped:
            X_group = group["X"]
            y_group = group["y"]
            linear_predictor = X_group @ beta
            log_denominator, inclusion_probs = _group_log_denominator_and_inclusion_probs(
                linear_predictor, group["successes"]
            )

            neg_loglike -= np.dot(y_group, linear_predictor) - log_denominator
            gradient -= X_group.T @ (y_group - inclusion_probs)

        return neg_loglike, gradient

    start_params = np.zeros(X.shape[1], dtype=float)
    result = minimize(
        fun=lambda beta: objective(beta)[0],
        x0=start_params,
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
    )

    if hasattr(result.hess_inv, "todense"):
        cov_matrix = np.asarray(result.hess_inv.todense(), dtype=float)
    else:
        cov_matrix = np.asarray(result.hess_inv, dtype=float)

    standard_errors = np.sqrt(np.clip(np.diag(cov_matrix), 0, None))
    z_scores = np.divide(
        result.x,
        standard_errors,
        out=np.full_like(result.x, np.nan),
        where=standard_errors > 0,
    )
    pvalues = 2 * (1 - norm.cdf(np.abs(z_scores)))

    return ConditionalLogitResults(
        params=pd.Series(result.x, index=column_names),
        bse=pd.Series(standard_errors, index=column_names),
        pvalues=pd.Series(pvalues, index=column_names),
        converged=bool(result.success),
        nobs=nobs,
        llf=-float(result.fun),
        llnull=llnull,
        model_name=model_name,
        dropped_groups=dropped_groups,
        used_groups=len(grouped),
    )


def run_logit_model(
    data,
    dependent,
    cat_controls=[],
    cont_controls=[],
    binary_vars=[],
    predictor="Has AI Skills",
    ref_category=None,
    get_vif=False,
    fixed_effect_group=None,
):
    """
    Runs a logistic regression model to predict work-from-home dependent status based on the specified predictor
    and control variables.

    Parameters:
    -----------
    controls : list of str
        A list of column names to be used as control variables. These variables will be converted to categorical
        variables and dummy variables will be created for them.
    predictor : str, default='Has AI Skills'
        The column name of the primary predictor variable.
    data : pd.DataFrame, default=data
        The DataFrame containing the data.
    ref_category : dict, optional
        A dictionary specifying the reference category for each control variable where keys are column names
        and values are the reference categories.

    Returns:
    --------
    logit_model : statsmodels.discrete.discrete_model.BinaryResultsWrapper
        The fitted logistic regression model.

    Example:
    --------
    >>> controls = ['region_column_name', 'industry_column_name']
    >>> model = run_logit_model(controls, predictor='Has AI Skills', data=data)
    >>> print(model.summary())
    """
    cat_controls = list(cat_controls)
    cont_controls = list(cont_controls)
    binary_vars = list(binary_vars)

    if fixed_effect_group is not None and fixed_effect_group in cat_controls:
        cat_controls = [control for control in cat_controls if control != fixed_effect_group]

    if "NAICS_2022_2_NAME" in cat_controls:
        industry_counts = data["NAICS_2022_2_NAME"].value_counts()
        small_industries = industry_counts[
            industry_counts < 50
        ].index  # Set a threshold (e.g., 10 observations)
        data["NAICS_2022_2_NAME"] = data["NAICS_2022_2_NAME"].replace(
            small_industries, "Other"
        )
    data = data.copy()
    # count nas in predictor
    print(f"Number of NAs in dependent: {data[dependent].isna().sum()}")
    data = data.dropna(subset=[dependent])
    if dependent == "wfh_wham" or dependent == "PARENTAL_LEAVE":
        data = data[data["YEAR"] != 2018]
    X = data[[predictor]]
    if binary_vars:
        for var in binary_vars:
            X = pd.concat([X, data[[var]]], axis=1)

    if cont_controls:
        for cc in cont_controls:
            X = pd.concat([X, data[[cc]]], axis=1)

    if cat_controls:
        for c in cat_controls:
            # Create categorical variables
            data[c] = data[c].astype("category")

            # Create dummy variables with specified reference category
            if ref_category and c in ref_category:
                # Ensure the reference category is present in the data
                if ref_category[c] in data[c].cat.categories:
                    data[c] = data[c].cat.reorder_categories(
                        [ref_category[c]]
                        + [
                            cat
                            for cat in data[c].cat.categories
                            if cat != ref_category[c]
                        ],
                        ordered=True,
                    )
                    dummies = pd.get_dummies(data[c], drop_first=True)
                # Create dummy variables
                else:
                    raise ValueError(
                        f"Reference category '{ref_category[c]}' not found in column '{c}'"
                    )
            else:
                dummies = pd.get_dummies(data[c], drop_first=True)
            X = pd.concat([X, dummies], axis=1)

    X = X.apply(pd.to_numeric, errors="coerce")

    y = data[dependent]
    keep_cols = [dependent]
    if fixed_effect_group is not None:
        keep_cols.append(fixed_effect_group)
    model_data = pd.concat([data[keep_cols], X], axis=1).dropna()
    X = model_data.drop(columns=keep_cols)
    y = model_data[dependent]
    y = pd.to_numeric(y, errors="coerce")

    X.columns = X.columns.astype(str)

    X = X.astype(float)
    y = y.astype(float)

    if fixed_effect_group is not None:
        groups = (
            model_data[fixed_effect_group]
            .astype(str)
            .to_numpy()
        )
    else:
        groups = None

    if fixed_effect_group is None:
        sm = _load_statsmodels()
        X = sm.add_constant(X)
    # print(X)
    # print(X.corr())

    if get_vif:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        print("Correlation Matrix")
        # create new df combine X and y for correlation matrix
        dummy_df = pd.concat([X, y], axis=1)
        dummy_corr_matrix = dummy_df.corr()
        # print any correlations above 0.7
        print(dummy_corr_matrix)

        print("VIF Results")
        vif_data = []
        for i in tqdm(range(X.shape[1]), desc="Calculating VIF"):
            vif = variance_inflation_factor(X.values, i)
            vif_data.append(vif)
        vif = pd.DataFrame()
        vif["features"] = X.columns
        vif["VIF"] = vif_data
        print(vif)

    # if error, continue to next model
    try:
        if fixed_effect_group is None:
            sm = _load_statsmodels()
            logit_model = sm.Logit(y, X).fit()
        else:
            logit_model = _fit_conditional_logit(
                X=X.to_numpy(),
                y=y.to_numpy(),
                groups=groups,
                column_names=X.columns,
                model_name=f"Conditional Logit ({fixed_effect_group} fixed effects)",
            )
    except Exception as error:
        print(error)
        return "Error"

    print(
        f"Logit fit: converged={logit_model.converged} "
        f"nobs={int(logit_model.nobs):,} "
        f"{predictor}={logit_model.params.get(predictor, np.nan):.4f}"
    )
    return logit_model


def save_model_summaries(models, save_path):
    summaries = []
    for model in models:
        summary = {
            "params": model.params,
            "pvalues": model.pvalues,
            "conf_int": model.conf_int(),
            "summary": model.summary().as_text(),
        }
        summaries.append(summary)

    with open(save_path, "wb") as f:
        pickle.dump(summaries, f)
