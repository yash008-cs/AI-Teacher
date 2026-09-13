# Machine Learning Fundamentals: Paradigms and Core Tasks

Welcome to these introductory study notes on Machine Learning (ML). Machine learning enables computational systems to discover patterns in historical data and make informed inferences or decisions without being explicitly programmed with static rule sets.

---

## 1. Supervised Learning

### Core Concept & Formulation
Supervised learning is the most common paradigm in applied artificial intelligence. In supervised learning, an algorithm learns a mapping function $f: X \rightarrow Y$ from an input feature space $X$ to an output target space $Y$, guided by a training dataset of labeled input-output pairs:
$$\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\}$$

Each example contains input features (such as student study hours, previous test scores, and attendance rates) paired with a ground-truth supervisory label (such as final exam score or pass/fail outcome).

### Learning Mechanism
During the training phase, the model generates predictions $\hat{y} = f(x; \theta)$ parameterized by internal weights $\theta$. A loss function $\mathcal{L}(y, \hat{y})$ measures the discrepancy between predicted values and actual ground-truth targets. Optimization algorithms, such as Gradient Descent and its adaptive variants, iteratively adjust parameters $\theta$ to minimize overall empirical risk across training batches.

### Key Strengths & Practical Considerations
- **High Predictive Power**: When abundant, high-quality labeled data exists, supervised models excel at producing accurate, reliable inferences.
- **Label Dependency**: Collecting and verifying ground-truth labels is often labor-intensive, costly, and subject to human annotation bias.
- **Evaluation Protocols**: Models must be evaluated on unseen validation and test partitions using metrics such as Accuracy, Precision, Recall, F1-Score, or Mean Squared Error to guard against overfitting.

---

## 2. Unsupervised Learning

### Core Concept & Formulation
In contrast to supervised paradigms, unsupervised learning operates on unlabeled datasets:
$$\mathcal{D} = \{x_1, x_2, \dots, x_N\}$$
Without external supervision or target outputs, the objective is to uncover inherent underlying structures, probability distributions, correlations, or natural groupings within the feature representations.

### Primary Sub-tasks
1. **Clustering**: Partitioning heterogeneous observations into cohesive, distinct clusters where items inside a cluster share strong similarity, while items across clusters are dissimilar. Notable algorithms include K-Means, DBSCAN, and Hierarchical Clustering.
2. **Dimensionality Reduction**: Projecting high-dimensional data points onto lower-dimensional manifolds while preserving maximal variance or topological neighborhoods. Techniques such as Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) facilitate computational efficiency, noise reduction, and intuitive 2D/3D visualization.
3. **Density Estimation & Anomaly Detection**: Modeling the probability density $p(x)$ to identify rare outliers or abnormal behavior in cybersecurity, fraud monitoring, and industrial machinery maintenance.

### Key Strengths & Practical Considerations
- **Label Independence**: Can exploit vast volumes of readily available, unannotated data.
- **Subjective Validation**: Evaluating model quality without ground truth is inherently challenging and often relies on heuristic metrics like silhouette scores or domain expert review.

---

## 3. Classification: Predicting Discrete Categories

### Definition & Objectives
Classification is a primary task within supervised learning where the target variable $y$ belongs to a discrete, finite set of categorical classes:
$$y \in \{C_1, C_2, \dots, C_K\}$$

- **Binary Classification**: Exactly two possible classes (e.g., Spam vs. Not Spam, Tumor Malignant vs. Benign).
- **Multiclass Classification**: More than two mutually exclusive classes (e.g., recognizing handwritten digits from 0 through 9).
- **Multilabel Classification**: An instance may simultaneously belong to multiple categories (e.g., tagging a research paper with both "Computer Science" and "Mathematics").

### Representative Algorithms & Decision Boundaries
- **Logistic Regression**: Uses the sigmoid activation function to map linear combinations of features into well-calibrated probabilities between 0 and 1.
- **Decision Trees & Random Forests**: Hierarchically partition feature space using orthogonal decision boundaries, offering intuitive rule interpretability and high resilience against feature scaling issues.
- **Support Vector Machines (SVM)**: Identify the maximum-margin hyperplane that separates opposing classes with optimal generalization buffer.
- **Neural Networks (MLPs, CNNs)**: Model complex non-linear decision boundaries through layered compositions of activation functions and tensor transformations.

### Real-World Applications
- Automated email spam filters and phishing detection systems.
- Medical diagnostic imaging for disease classification.
- Customer sentiment analysis in educational feedback reviews.

---

## 4. Regression: Predicting Continuous Quantities

### Definition & Objectives
Regression is a supervised learning task where the target output variable $y$ is a continuous numerical real value:
$$y \in \mathbb{R}$$
The goal is to model trend lines or multidimensional response surfaces that predict quantities, rates, durations, or monetary values given descriptive predictors.

### Representative Algorithms
- **Linear Regression**: Assumes an additive relationship where predictions are linear combinations of features weighted by learnable coefficients:
$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_D x_D$$
- **Regularized Regression (Ridge & Lasso)**: Penalizes large model weights ($L_2$ or $L_1$ penalties) to curb overfitting and perform automatic feature selection.
- **Support Vector Regression (SVR)**: Constructs an $\epsilon$-insensitive tube where errors smaller than a threshold incur zero loss penalty.
- **Gradient Boosted Trees (XGBoost, LightGBM)**: Sequentially train ensembles of shallow decision trees where each new tree fits the residual errors of prior iterations.

### Evaluation Metrics
- **Mean Squared Error (MSE)**: $\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$, heavily penalizes large outlier errors.
- **Mean Absolute Error (MAE)**: $\frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$, provides robust linear penalties less sensitive to anomalies.
- **Root Mean Squared Error (RMSE)**: Preserves original target units for intuitive interpretation.
- **Coefficient of Determination ($R^2$)**: Quantifies the proportion of target variance explained by the explanatory features.

### Real-World Applications
- Real estate valuation and housing market price estimation.
- Student test score trajectory projections and study pace estimation.
- Algorithmic energy demand forecasting for regional electrical grids.

---

## 5. Comparative Summary: Classification vs. Regression

| Dimension | Classification | Regression |
| :--- | :--- | :--- |
| **Output Type** | Discrete categories / class labels | Continuous numerical real values |
| **Prediction Example** | "Will the student pass the exam? (Yes/No)" | "What numerical score (0-100) will the student achieve?" |
| **Primary Algorithms** | Logistic Regression, Decision Trees, SVM | Linear Regression, Ridge, SVR, Gradient Boosting |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1, ROC-AUC | MSE, MAE, RMSE, $R^2$ Score |
| **Decision Surface** | Separating boundaries between classes | Continuous fitting curve or multidimensional hypersurface |
