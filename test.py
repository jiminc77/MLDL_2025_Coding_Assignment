import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 1. 데이터 로드
df = pd.read_csv(r"C:\Users\jimin\Python\Exercise\MLDL_2025\train.csv")
X_raw = df.drop(columns=["Y"])
y = df["Y"].to_numpy()

# -----------------------
# 2. 상호작용 및 비율 피처 추가
def add_interactions_df(X_df):
    X_new = X_df.copy()
    new_feature_names = []

    interactions = [(4, 9), (3, 9), (10, 16), (11, 16), (4, 10), (6, 8)]
    for i, j in interactions:
        name = f"X{i}*X{j}"
        X_new[name] = X_df.iloc[:, i] * X_df.iloc[:, j]
        new_feature_names.append(name)

    important_features = [17, 6, 16, 3, 11, 5] 
    for i in range(len(important_features) - 1):
        for j in range(i + 1, len(important_features)):
            fi, fj = important_features[i], important_features[j]
            name = f"X{fi}/X{fj}"
            X_new[name] = X_df.iloc[:, fi] / (X_df.iloc[:, fj] + 1e-8)
            new_feature_names.append(name)

    return X_new, new_feature_names

X_full_df, _ = add_interactions_df(X_raw)
X_full = X_full_df.to_numpy()
all_feature_names = list(X_full_df.columns)

# -----------------------
# 3. 상관계수 계산 (NaN 고려 + 절댓값 추가)
def compute_correlations_with_nan(X, y, feature_names):
    pearson_corrs = []
    spearman_corrs = []

    y_series = pd.Series(y)
    y_rank = y_series.rank().to_numpy()

    for i in range(X.shape[1]):
        xi = X[:, i]
        xi_series = pd.Series(xi)

        valid = ~(xi_series.isna() | y_series.isna())
        xi_clean = xi_series[valid].to_numpy()
        y_clean = y_series[valid].to_numpy()

        # Pearson
        if len(xi_clean) < 2:
            p_corr = np.nan
        else:
            p_corr = np.corrcoef(xi_clean, y_clean)[0, 1]
        pearson_corrs.append(p_corr)

        # Spearman
        xi_rank = pd.Series(xi_clean).rank().to_numpy()
        y_rank_clean = pd.Series(y_clean).rank().to_numpy()

        if len(xi_clean) < 2 or np.std(xi_rank) == 0 or np.std(y_rank_clean) == 0:
            s_corr = np.nan
        else:
            cov = np.cov(xi_rank, y_rank_clean)[0, 1]
            s_corr = cov / (np.std(xi_rank) * np.std(y_rank_clean))
        spearman_corrs.append(s_corr)

    df_corr = pd.DataFrame({
        "Feature": feature_names,
        "Pearson": pearson_corrs,
        "Spearman": spearman_corrs
    })
    df_corr["|Pearson|"] = df_corr["Pearson"].abs()
    df_corr["|Spearman|"] = df_corr["Spearman"].abs()
    return df_corr

df_corr = compute_correlations_with_nan(X_full, y, all_feature_names)

# -----------------------
# 4. 절댓값 기준 정렬 결과 출력 함수 (차트 X, 텍스트 O)
def print_sorted_abs_correlations(df_corr, method="|Pearson|"):
    sorted_df = df_corr.sort_values(method, ascending=False)

    print(f"\n🧪 {method} 기준 오름차순 정렬:")
    print(sorted_df[["Feature", "Pearson", "Spearman"]].to_string(index=False))

# -----------------------
# 5. 실행
print_sorted_abs_correlations(df_corr, method="|Pearson|")
print_sorted_abs_correlations(df_corr, method="|Spearman|")
