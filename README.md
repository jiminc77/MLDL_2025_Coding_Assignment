## 1. Baseline 코드 분석

---

### 1.1 구현 개요

Baseline 코드에서는 기본적인 모델 구현을 위한 템플릿만 제공. 데이터 로드 및 분할, 모델 클래스 정의, 예측 및 제출 파일 생성을 위한 기본 구조 포함.

```python
class Model:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
```

### 1.2 데이터 전처리

Baseline 코드에서는 다음과 같은 기본적인 데이터 전처리만 수행함:

- CSV 파일에서 데이터 로드
- ID 컬럼 제거 (존재하는 경우)
- 특성(X)과 레이블(y) 분리
- Train/test split (테스트 비율 20%, random seed 32)

```python
# Load dataset
df = pd.read_csv('/kaggle/input/mldl-2025/train.csv')

# Drop ID column if exists
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# Split features and label
X = df.drop(columns=['Y']).values
y = df['Y'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
```

## 2. Ver 1: Logistic Regression + L2 Regularization

---

### 2.1 구현 개요

첫 번째 버전에서는 logistic regression 모델을 직접 구현하고 L2 regularization을 적용함. 데이터의 분포는 크게 고려하지 않고, missing value 처리와 기본적인 normalization만 수행함.

### 2.2 데이터 전처리

Baseline 대비 다음과 같은 전처리 과정이 추가됨:

- Missing value를 median으로 대체
- Standardization 적용 (평균 0, 표준편차 1)
- Stratified sampling 적용하여 훈련/검증 데이터 분할

```python
# Split features and label
features = df.drop(columns=['Y'])
X = features.fillna(features.median(numeric_only=True)).values
y = df['Y'].values

# Train/test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=32, stratify=y)

# Standardize
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0) + 1e-8

def standardize(a): return (a - mu) / sigma

X_train = standardize(X_train)
X_valid = standardize(X_valid)
```

### 2.3 모델 구현

Logistic regression 모델을 직접 구현하고 L2 regularization을 적용함:

```python
class Model:
    def __init__(self, lr=0.05, epochs=4000, reg=0.0):
        self.lr, self.epochs, self.reg = lr, epochs, reg

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def _add_bias(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):
        X_b = self._add_bias(X)
        n_features = X_b.shape[1]
        self.w = np.zeros(n_features)

        for _ in range(self.epochs):
            z = X_b @ self.w
            h = self._sigmoid(z)
            grad = (X_b.T @ (h - y)) / len(y)
            # L2 Normalization
            grad[1:] += self.reg * self.w[1:] / len(y)
            self.w -= self.lr * grad

    def predict_proba(self, X):
        X_b = self._add_bias(X)
        return self._sigmoid(X_b @ self.w)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
```

**주요 구현 특징:**

- Bias 항을 추가하는 메서드 구현
- Gradient descent를 통한 weight update
- L2 regularization 적용
- Prediction threshold 0.5 이상을 positive class로 분류

### 2.4 하이퍼파라미터

다음과 같은 hyperparameter를 사용함:

- Learning rate(lr): 0.05
- Epochs: 4000
- Regularization coefficient(reg): 0.01

### 2.5 성능 평가

Validation dataset에 대한 accuracy는 56.31%로, 목표치에 크게 미치지 못하는 결과를 보임.

### 2.6 한계점 및 개선 방향

첫 번째 버전의 주요 한계점:

- 단순한 linear model로 복잡한 데이터 패턴 포착 어려움
- Feature engineering 부재로 모델의 표현력 제한
- Hyperparameter optimization이 충분히 이루어지지 않음

이러한 한계점을 극복하기 위해 데이터의 특성을 더 깊이 분석하고, 모델의 complexity를 높이는 방향으로 개선이 필요했음.

## 3. Ver 2: Feature Engineering + Bagging

---

### 3.1 구현 개요

두 번째 버전에서는 데이터 특성 분석을 통해 중요 feature를 식별하고, 이를 기반으로 feature engineering을 수행함. 또한, 모델의 stability와 performance를 향상시키기 위해 bagging ensemble 기법을 도입함.

### 3.2 데이터 분석 및 Feature Engineering

데이터 특성 분석을 통해 상위 중요 특성 6개를 식별하고, 이를 기반으로 추가 feature를 생성함:

```python
top_idx = np.array([11, 13, 15, 10, 6, 17]) 
X_top = X_base[:, top_idx]

X_sq = X_top ** 2

pairs = [(i, j) for i in range(6) for j in range(i+1, 6)]
X_cross = np.column_stack([X_top[:, i] * X_top[:, j] for i, j in pairs])

X_ext = np.hstack([X_base, X_sq, X_cross])
```

**주요 feature engineering:**

- 상위 중요 특성 6개 선정 (X17, X6, X13, X10, X16, X1)
- 선정된 특성들의 squared term 추가 (6개)
- 선정된 특성들의 interaction term 추가 (15개)
- 총 21개의 추가 feature 생성

이를 통해 logistic regression 모델이 non-linear decision boundary를 학습할 수 있도록 모델의 complexity를 증가시킴.

### 3.3 개선된 Logistic Regression 모델

기존 logistic regression 모델에 class weight를 도입하여 imbalanced data 처리 능력을 향상시킴:

```python
class LR:
    def __init__(self, lr=0.04, epochs=5000, reg=0.02, alpha=1.5):
        self.lr, self.epochs, self.reg, self.alpha = lr, epochs, reg, alpha

    def _sigmoid(self, z): return 1 / (1 + np.exp(-z))
    def _bias(self, X): return np.hstack((np.ones((X.shape[0], 1)), X))

    def fit(self, X, y):
        Xb, m = self._bias(X), len(y)
        self.w = np.zeros(Xb.shape[1])
        # class weights: w_pos = alpha (>1), w_neg = 1
        w_vec = np.where(y == 1, self.alpha, 1.0)
        for _ in range(self.epochs):
            h = self._sigmoid(Xb @ self.w)
            grad = (Xb.T @ (w_vec * (h - y))) / m
            grad[1:] += self.reg * self.w[1:] / m
            self.w -= self.lr * grad
```

**주요 개선 사항:**

- Class weight(alpha) 도입: positive class에 더 높은 가중치 부여
- Weight vector(`w_vec`)를 통한 class별 가중치 적용
- 코드 간소화 및 최적화

### 3.4 Bagging Ensemble 구현

모델의 stability와 performance를 향상시키기 위해 bagging ensemble 기법을 도입함:

```python
class Bagging:
    def __init__(self, n=75, samp_ratio=0.9,
                 feat_ratio=0.7, base_kw=None, thr=0.5):
        self.n, self.sr, self.fr, self.bkw, self.thr = n, samp_ratio, feat_ratio, base_kw or {}, thr

    def _boot_idx(self, y):
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        m = int(len(y) * self.sr / 2)
        idx = np.concatenate([np.random.choice(pos, m, True),
                              np.random.choice(neg, m, True)])
        return np.random.permutation(idx)

    def fit(self, X, y):
        self.models, self.feat_sets = [], []
        d = X.shape[1]
        for _ in range(self.n):
            fi = np.random.choice(d, int(d * self.fr), replace=False)
            ri = self._boot_idx(y)
            m = LR(**self.bkw)
            m.fit(X[ri][:, fi], y[ri])
            self.models.append(m); self.feat_sets.append(fi)

    def predict(self, X):
        votes = np.zeros((X.shape[0], self.n))
        for i, (m, f) in enumerate(zip(self.models, self.feat_sets)):
            votes[:, i] = m.predict(X[:, f], thr=self.thr)
        return (votes.mean(1) >= 0.5).astype(int)

```

**주요 구현 특징:**

- **Base learner 구성**
    - `LR` 기반 모델 75개 생성
    - 각 모델은 독립적으로 학습됨
- **Bootstrap + 클래스 균형 샘플링**
    - 전체 샘플 중 90% 비율의 데이터 사용
    - 클래스 균형 유지: positive/negative 각각 동일 개수 추출
    - 중복 허용(random sampling with replacement)
- **Feature subsampling**
    - 전체 feature 중 70% 무작위 선택
    - 모델마다 feature subset 상이
- **Ensemble 예측**
    - 각 모델 예측값 평균 → 0.5 이상이면 1, 아니면 0

### 3.5 하이퍼파라미터

**Bagging ensemble 모델의 hyperparameter:**

- Ensemble size(`n`): 75
- Sampling ratio(`samp_ratio`): 0.9
- Feature ratio(`feat_ratio`): 0.7
- Prediction threshold(`thr`): 0.5

**Logistic regression 모델의 hyperparameter:**

- Learning rate(`lr`): 0.04
- Epochs: 5000
- Regularization coefficient(`reg`): 0.02
- Class weight(`alpha`): 1.5

### 3.6 성능 평가

Validation dataset에 대한 accuracy는 **68.50%**로, 첫 버전(**56.31%**)에 비해 **12.19%p** 향상된 결과를 보임.

### 3.7 개선 효과 및 한계점

**개선 효과:**

- Feature engineering을 통한 모델 expressiveness 향상
- Bagging ensemble을 통한 모델 stability 향상
- Class weight 도입으로 imbalanced data 처리 개선
- Hyperparameter optimization을 통한 성능 향상

**한계점:**

- 여전히 logistic regression 기반으로 복잡한 패턴 포착에 한계 존재
- Hard voting 방식은 noise에 취약
- Bootstrap bias를 충분히 평균화하지 못함
- 더 높은 성능을 위해서는 non-linearity가 강한 모델이 필요함

## 4. Final version: Random Forest

---

### 4.1 구현 개요

최종 버전에서는 non-linearity가 강한 데이터셋에서 복잡한 패턴을 더 효과적으로 포착하기 위해 decision tree 기반의 random forest 모델을 직접 구현함. Tree 모델의 특성에 맞게 feature engineering을 수행하고, 다양한 최적화 기법을 적용함.

### 4.2 데이터 전처리 및 Feature Engineering

기본적인 전처리는 이전 버전과 유사하게 수행하되, tree 모델에 적합한 feature engineering을 추가함:

```python
def add_interactions(X):
    X_new = X.copy()

    # Top correlated pairs from the analysis
    interactions = [
        (4, 9),   # 0.787 correlation
        (3, 9),   # 0.755 correlation
        (10, 16), # 0.695 correlation
        (11, 16), # 0.677 correlation
        (4, 10),  # 0.667 correlation
        (6, 8),   # 0.588 correlation
    ]

    # Multiplication interactions
    for i, j in interactions:
        X_new = np.column_stack([X_new, X[:, i] * X[:, j]])

    # Ratio features for features with high individual importance
    important_features = [11, 13, 15, 10, 6, 17]
    for i in range(len(important_features)-1):
        for j in range(i+1, len(important_features)):
            fi, fj = important_features[i], important_features[j]
            ratio = X[:, fi] / (X[:, fj] + 1e-8)
            X_new = np.column_stack([X_new, ratio])

    return X_new
```

**주요 feature engineering:**

- Correlation analysis를 통해 선정된 feature 쌍 간 multiplication interaction 추가 (6쌍)
- 중요도가 높은 feature 간 ratio feature 추가
- Division by zero 오류 방지를 위한 작은 값(1e-8) 추가

**이전 버전과의 차이점:**

- Correlation analysis를 기반으로 한 더 체계적인 feature 선정
- Ratio feature 추가로 tree 모델의 split efficiency 향상
- Feature 간 관계를 더 명시적으로 모델링

### 4.3 Random Forest 모델 구현

Decision tree 기반의 random forest 모델을 직접 구현함:

```python
class Model:
    def __init__(self):
        # Random Forest parameters
        self.n_estimators = 400
        self.max_depth = 20
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.max_features = 'sqrt'  # sqrt of total features
        self.bootstrap = True

        self.trees = []
        self.feature_indices = []

        self.oob_indices = []
        self.patience = 10
        self.best_oob = -1
        self.no_improve = 0
```

**주요 구현 특징:**

1. **Gini impurity와 information gain 계산:**
    
    ```python
    def _gini_impurity(self, y):
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        p = np.sum(y == 1) / len(y)
        return 2 * p * (1 - p)
    
    def _information_gain(self, y, left_y, right_y):
        """Calculate information gain"""
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
    ```
    
    - **Gini impurity 기반 노드 purity 측정**
        - 클래스 비율 $p$를 이용해 $2p(1−p)$ 방식으로 계산 (0에 가까울수록 pure)
    - **Information Gain 계산**
        - 부모 노드와 자식 노드의 지니 지수를 가중 평균하여 Information Gain 계산
        - 분할이 유의미하지 않으면 이득은 0에 수렴
2. **Decision tree 구축 알고리즘:**
    
    ```python
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
    
        # Stopping criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            # Return leaf node with majority class
            return {'leaf': True, 'prediction': np.round(np.mean(y))}
    
        # Feature subsampling
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        else:
            max_features = n_features
    
        feature_indices = np.random.choice(n_features, max_features, replace=False)
    
        # Find best split
        best_gain = -1
        best_feature = None
        best_threshold = None
    
        for feature_idx in feature_indices:
            # Try multiple threshold candidates
            thresholds = np.percentile(X[:, feature_idx], [10, 25, 50, 75, 90])
    
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
    
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
    
                gain = self._information_gain(y, y[left_mask], y[right_mask])
    
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
    
        # If no good split found
        if best_feature is None:
            return {'leaf': True, 'prediction': np.round(np.mean(y))}
    
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
    
        # Build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
    
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    ```
    
    - **재귀적 분할 기반 트리 구성**
        - `max_depth`, `min_samples_split` 등 stopping 조건 명시
    - **특성 서브샘플링 (`max_features` = `sqrt`)**
        - √(전체 특성 수)만큼 랜덤으로 추출하여 분할 기준 탐색
    - **Percentile 기반 threshold 후보**
        - 10, 25, 50, 75, 90% 사용하여 overfitting 방지
    - **최적 information gain 기준 split**
        - 유효한 분할 중 가장 높은 information gain을 제공하는 특성과  threshold 선택
3. **모델 학습 및 OOB 평가:**
    
    ```python
    def fit(self, X, y):
        n_samples = X.shape[0]
    
        for i in range(self.n_estimators):
            # Bootstrap sampling
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
    
            # Build tree
            tree = self._build_tree(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
    
            # Progress update
            if (i + 1) % 10 == 0:
                oob_pred = self._get_oob_predictions(X, i + 1)
                oob_acc = np.mean(oob_pred == y)
                print(f"[{i + 1:3d}] OOB Accuracy = {oob_acc*100:.2f}%")
    
                ### Early stopping
                if oob_acc > self.best_oob + 1e-6:
                    self.best_oob = oob_acc
                    self.no_improve = 0
                else:
                    self.no_improve += 1
                if self.no_improve >= self.patience:
                    print("Early-stop triggered")
                    break
    ```
    
    - **Bootstrap 샘플링 기반 트리 훈련**
        - 각 트리는 중복 허용 랜덤 샘플을 사용해 개별적으로 훈련
    - **OOB (Out-of-Bag) 평가 구현**
        - Bootstrap에 포함되지 않은 샘플로 트리 성능 평가 (교차검증 대체)
    - **Early Stopping**
        - OOB accuracy가 10 epoch 동안 개선되지 않으면 학습 중단
4. **OOB 예측 및 최종 예측:**
    
    ```python
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
        probas = self.predict_proba(X)
        return (probas > 0.5).astype(int)
    ```
    
    - **OOB 예측 집계**
        - 각 트리에서 OOB 샘플에 대한 예측을 평균하여 다수결 투표 형태로 예측 생성
    - **확률 기반 예측**
        - `predict_proba` 함수는 각 트리의 평균 예측값으로 클래스 1의 확률 추정
    - **Binary classification을 위한 최종 예측**
        - 0.5를 기준으로 thresholding하여 0 또는 1 클래스 반환

### 4.4 주요 개선 사항

**이전 버전 대비 주요 개선 사항:**

1. **모델 아키텍처 변경**:
    - Logistic regression에서 decision tree 기반 random forest로 변경
    - Non-linearity가 강한 데이터에 더 적합한 모델 선택
2. **Split 기준 최적화**:
    - Information gain 기반 split 기준 사용
    - 다양한 percentile(10%, 25%, 50%, 75%, 90%)을 threshold 후보로 사용
3. **OOB 샘플 활용**:
    - Bootstrap sampling 시 제외된 샘플을 validation에 활용
    - 모델 학습 중 performance monitoring 및 early stopping 기준으로 활용
4. **Early stopping 구현**:
    - OOB accuracy가 10회 연속으로 개선되지 않으면 학습 중단
    - Overfitting 방지 및 학습 효율성 향상
5. **Soft voting 방식 사용:**
    - 각 tree의 예측 확률을 평균 후 최종 예측 결정
    - Hard voting 대비 noise에 robust한 예측 방식
6. **Feature engineering 개선**:
    - Correlation analysis 기반 interaction feature 추가
    - 중요 feature 간 ratio feature 추가로 tree 모델의 split efficiency 향상

### 4.5 하이퍼파라미터

**Random forest 모델의 hyperparameter:**

```python
max_depth_list = [5, 10, 15, 20]
min_samples_split_list = [2, 5, 10]
min_samples_leaf_list = [1, 2, 4, 8]

for md in max_depth_list:
    for mss in min_samples_split_list:
        for msl in min_samples_leaf_list:
            model = Model()
            model.n_estimators = 300
            model.max_depth = md
            model.min_samples_split = mss
            model.min_samples_leaf = msl
            model.max_features = 'sqrt'
            model.bootstrap = True
            
            print(f"\n--- Parameters: max_depth={md}, min_samples_split={mss}, min_samples_leaf={msl} ---")
            
            model.fit(X_train, y_train)

            val_preds = model.predict(X_test)
            val_accuracy = np.mean(val_preds == y_test)
            print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
```

- **Grid Search를 통해 최적값 산출**
    - Number of trees(`n_estimators`): 400
    - Maximum depth(`max_depth`): 20
    - Minimum samples split(`min_samples_split`): 2
    - Minimum samples leaf(`min_samples_leaf`): 1
    - Feature selection method(`max_features`): 'sqrt'
    - Bootstrap sampling(`bootstrap`): True
    - Early stopping patience(`patience`): 10

### 4.6 성능 평가

Validation set에 대한 accuracy는 **75.88%**로, 두 번째 버전(**68.50%**)에 비해 **7.38%p** 향상된 결과를 보임.

## 5. 성능 비교 및 분석

### 5.1 성능 변화 요약

**각 버전별 validation accuracy 변화:**

- Baseline: 구현 없음
- V**er 1** (Logistic Regression): 56.31%
- **Ver 2** (Logistic Regression + Bagging): 68.50% (+12.19%p)
- **Final version** (Random Forest): 75.88% (+7.38%p)

총 성능 향상: **+19.57%p** (Ver 1 대비 Final_version)

### 5.2 실패 요인 및 한계점

1. **초기 데이터 특성 분석 부족 (Ver 1)**:
    - Ver 1에서 데이터의 non-linear 특성을 고려하지 않음
    - 데이터 탐색 없이 바로 모델링 시도
2. **Logistic Regression 모델의 Expressiveness 한계 (Ver 2)**:
    - Linear model로 복잡한 non-linear pattern 포착 어려움
    - Feature engineering으로 일부 보완했으나 근본적 한계 존재
3. **더욱 강력한 Feature Engineering, 비선형 패턴 해석 모델의 필요성 (Final Ver)**:
    - 아직도 정확도가 75% 정도
    - 더욱 강력한 Engineering Technique이 필요

## 6. 결론

Binary classification 문제를 해결하기 위한 ML 모델 개발 과정을 단계별로 고찰함. 초기 logistic regression 모델에서 시작하여 feature engineering, bagging ensemble, 그리고 최종적으로 random forest 모델까지 점진적인 개선을 통해 성능을 향상시킴.

**주요 발견점:**

1. 데이터의 특성을 이해하고 이에 맞는 모델을 선택하는 것이 중요함
2. Feature engineering은 모델 performance 향상에 큰 영향을 미침
3. Ensemble 기법은 단일 모델 대비 stability와 performance를 크게 향상시킴
4. OOB 샘플을 활용한 monitoring과 early stopping은 효율적인 모델 학습에 도움이 됨
5. Soft voting 방식은 hard voting 대비 noise에 robust한 prediction을 제공함

**향후 개선 방향:**

1. 더 다양한 feature engineering 기법 탐색
2. Hyperparameter optimization을 위한 효율적인 방법론 적용
3. 다양한 ensemble 기법(Gradient boosting 등) 비교 분석
4. Imbalanced data 처리를 위한 추가적인 기법 적용