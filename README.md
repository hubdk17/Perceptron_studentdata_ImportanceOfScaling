# Perceptron_studentdata_ImportanceOfScaling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import MinMaxScaler

# --- Generate simple dataset ---
np.random.seed(42)
cgpa_placed = np.random.uniform(7.5, 10.0, 120)
resume_placed = np.random.uniform(60.0, 70.0, 120)
cgpa_not_placed = np.random.uniform(6.0, 7.5, 80)
resume_not_placed = np.random.uniform(50.0, 62.0, 80)

data = pd.DataFrame({
    'cgpa': np.concatenate([cgpa_placed, cgpa_not_placed]),
    'resume_score': np.concatenate([resume_placed, resume_not_placed]),
    'placed': np.concatenate([np.ones(120, dtype=int), np.zeros(80, dtype=int)])
}).sample(frac=1, random_state=1).reset_index(drop=True)

print(data.head())

# --- Train Perceptron ---
X = data[['cgpa', 'resume_score']]
y = data['placed']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

model = Perceptron(eta0=0.1, max_iter=1000, random_state=0, tol=1e-3)
model.fit(X_scaled, y)

# --- Extract weights in scaled space and map to original scale ---
W1, W2 = model.coef_[0]
b = model.intercept_[0]

x1_min, x1_max = scaler.data_min_[0], scaler.data_max_[0]
x2_min, x2_max = scaler.data_min_[1], scaler.data_max_[1]
x1_range = x1_max - x1_min
x2_range = x2_max - x2_min

W1_orig = W1 / x1_range
W2_orig = W2 / x2_range
b_orig = b - (W1 * x1_min / x1_range) - (W2 * x2_min / x2_range)

# --- Plot decision boundary ---
plt.figure(figsize=(8, 6))
plt.scatter(X['cgpa'][y == 0], X['resume_score'][y == 0], color='red', label='Not Placed')
plt.scatter(X['cgpa'][y == 1], X['resume_score'][y == 1], color='blue', label='Placed')

x_line = np.linspace(X['cgpa'].min(), X['cgpa'].max(), 100)
y_line = (-W1_orig * x_line - b_orig) / W2_orig

plt.plot(x_line, y_line, 'k--', linewidth=2, label='Decision Boundary')
plt.xlabel('CGPA')
plt.ylabel('Resume Score')
plt.title('Perceptron Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()
