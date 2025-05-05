# Loading and preparing the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Loading dataset
df = pd.read_csv('Breast_cancer_data.csv')

# Preview data
print(df.head())

# Encode target: M = 1 (Malignant), B = 0 (Benign)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Drop any non-feature columns (like ID or unnamed columns)
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Linear SVM

# Linear kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred_linear = svm_linear.predict(X_test_scaled)
print("Linear SVM Results:")
print(confusion_matrix(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))

# Train a Non-linear SVM (RBF Kernel)

# RBF kernel
svm_rbf = SVC(kernel='rbf', gamma='scale')
svm_rbf.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred_rbf = svm_rbf.predict(X_test_scaled)
print("RBF SVM Results:")
print(confusion_matrix(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))

# Reduce features to 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_train_scaled)

# Fit SVM on reduced data
svm_2d = SVC(kernel='linear')
svm_2d.fit(X_reduced, y_train)

# Plot decision boundary
def plot_svm_decision_boundary(clf, X, y):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title("SVM Decision Boundary (2D PCA)")
    plt.show()

plot_svm_decision_boundary(svm_2d, X_reduced, y_train)

# Define parameter grid for linear and RBF kernel
param_grid_rbf = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf']
}

param_grid_linear = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear']
}

# Combine parameter grids
combined_grid = [param_grid_rbf, param_grid_linear]

# Initialize SVM
svc = SVC()

# GridSearchCV setup
grid_search = GridSearchCV(
    svc,
    combined_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit the grid search
grid_search.fit(X, y)

print("✅ Best Parameters Found:")
print(grid_search.best_params_)
print(f"✅ Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

# Use the best estimator from GridSearch
best_svm = grid_search.best_estimator_

# 5-fold cross-validation accuracy
cv_scores = cross_val_score(best_svm, X, y, cv=5, scoring='accuracy')

print("✅ Cross-validation Accuracy Scores:", cv_scores)
print(f"✅ Mean Accuracy: {cv_scores.mean():.4f}")
print(f"✅ Standard Deviation: {cv_scores.std():.4f}")






