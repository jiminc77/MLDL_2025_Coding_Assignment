# K-Fold cross-validation script using the custom Random Forest model
import numpy as np
import pandas as pd

# ---- Data Loading ----
train_df = pd.read_csv('train.csv')
if 'ID' in train_df.columns:
    train_df = train_df.drop(columns=['ID'])

X_full = train_df.drop(columns=['Y'])
y_full = train_df['Y'].values

# Fill missing values and standardize
X_filled = X_full.fillna(X_full.median(numeric_only=True))
X_values = X_filled.values
mu = X_values.mean(axis=0)
sigma = X_values.std(axis=0) + 1e-8
std = lambda a: (a - mu) / sigma
X_values = std(X_values)

# ---- Feature Engineering ----
def add_interactions(X):
    X_new = X.copy()
    interactions = [
        (4, 9),
        (3, 9),
        (10, 16),
        (11, 16),
        (4, 10),
        (6, 8),
    ]
    for i, j in interactions:
        X_new = np.column_stack([X_new, X[:, i] * X[:, j]])
    important_features = [11, 13, 15, 10, 6, 17]
    for i in range(len(important_features) - 1):
        for j in range(i + 1, len(important_features)):
            fi, fj = important_features[i], important_features[j]
            ratio = X[:, fi] / (X[:, fj] + 1e-8)
            X_new = np.column_stack([X_new, ratio])
    return X_new

X_values = add_interactions(X_values)

class Model:
    def __init__(self):
        self.n_estimators = 400
        self.max_depth = 20
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.max_features = 'sqrt'
        self.bootstrap = True
        self.trees = []
        self.feature_indices = []
        self.oob_indices = []
        self.patience = 10
        self.best_oob = -1
        self.no_improve = 0

    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0
        p = np.sum(y == 1) / len(y)
        return 2 * p * (1 - p)

    def _information_gain(self, y, left_y, right_y):
        n = len(y)
        if n == 0:
            return 0
        n_left = len(left_y)
        n_right = len(right_y)
        parent_gini = self._gini_impurity(y)
        left_gini = self._gini_impurity(left_y)
        right_gini = self._gini_impurity(right_y)
        weighted_gini = (n_left / n) * left_gini + (n_right / n) * right_gini
        return parent_gini - weighted_gini
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return {'leaf': True, 'prediction': np.round(np.mean(y))}
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        else:
            max_features = n_features
        feature_indices = np.random.choice(n_features, max_features, replace=False)
        best_gain = -1
        best_feature = None
        best_threshold = None
        for feature_idx in feature_indices:
            thresholds = np.percentile(X[:, feature_idx], [10, 25, 50, 75, 90])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if (
                    np.sum(left_mask) < self.min_samples_leaf
                    or np.sum(right_mask) < self.min_samples_leaf
                ):
                    continue
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        if best_feature is None:
            return {'leaf': True, 'prediction': np.round(np.mean(y))}
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree,
        }
    def _predict_tree(self, tree, X):
        if tree['leaf']:
            return np.full(len(X), tree['prediction'])
        predictions = np.zeros(len(X))
        left_mask = X[:, tree['feature']] <= tree['threshold']
        right_mask = ~left_mask
        if np.sum(left_mask) > 0:
            predictions[left_mask] = self._predict_tree(tree['left'], X[left_mask])
        if np.sum(right_mask) > 0:
            predictions[right_mask] = self._predict_tree(tree['right'], X[right_mask])
        return predictions

    def fit(self, X, y):
        n_samples = X.shape[0]
        for i in range(self.n_estimators):
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
                oob_idx = np.setdiff1d(np.arange(n_samples), indices)
                self.oob_indices.append(oob_idx)
            else:
                X_bootstrap = X
                y_bootstrap = y
                self.oob_indices.append(np.arange(n_samples))
            tree = self._build_tree(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            if (i + 1) % 10 == 0:
                oob_pred = self._get_oob_predictions(X, i + 1)
                oob_acc = np.mean(oob_pred == y)
                print(f"[{i + 1:3d}] OOB Accuracy = {oob_acc*100:.2f}%")
                if oob_acc > self.best_oob + 1e-6:
                    self.best_oob = oob_acc
                    self.no_improve = 0
                else:
                    self.no_improve += 1
                if self.no_improve >= self.patience:
                    print('Early-stop triggered')
                    break
    def _get_oob_predictions(self, X, n_trees):
        n_samples = X.shape[0]
        oob_votes = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)
        for t in range(n_trees):
            idx = self.oob_indices[t]
            if idx.size == 0:
                continue
            preds = self._predict_tree(self.trees[t], X[idx])
            oob_votes[idx] += preds
            oob_counts[idx] += 1
        mask = oob_counts > 0
        oob_final = np.zeros(n_samples, dtype=int)
        oob_final[mask] = (oob_votes[mask] / oob_counts[mask] > 0.5).astype(int)
        return oob_final

    def predict_proba(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        for tree in self.trees:
            predictions += self._predict_tree(tree, X)
        return predictions / len(self.trees)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
# ---- K-Fold Utilities ----
def k_fold_indices(n_samples, k, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    fold_sizes = [n_samples // k] * k
    for i in range(n_samples % k):
        fold_sizes[i] += 1
    folds = []
    current = 0
    for size in fold_sizes:
        folds.append(indices[current : current + size])
        current += size
    return folds

def cross_val_score(X, y, params, k=5):
    folds = k_fold_indices(len(X), k)
    scores = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        model = Model()
        for key, value in params.items():
            setattr(model, key, value)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        score = np.mean(preds == y[val_idx])
        scores.append(score)
    return np.mean(scores)
# ---- Hyperparameter Search ----
param_grid = {
    'max_depth': [15, 20],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
}

best_score = -1
best_params = None
for md in param_grid['max_depth']:
    for mss in param_grid['min_samples_split']:
        for msl in param_grid['min_samples_leaf']:
            params = {
                'max_depth': md,
                'min_samples_split': mss,
                'min_samples_leaf': msl,
            }
            score = cross_val_score(X_values, y_full, params, k=5)
            print(
                f"Params depth={md}, split={mss}, leaf={msl} -> CV Acc {score*100:.2f}%"
            )
            if score > best_score:
                best_score = score
                best_params = params

print('\nBest params:', best_params)
print('Best CV Accuracy: {:.2f}%'.format(best_score * 100))

# ---- Train Final Model ----
final_model = Model()
for k, v in best_params.items():
    setattr(final_model, k, v)
final_model.fit(X_values, y_full)
