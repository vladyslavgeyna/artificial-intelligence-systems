import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def generate_data(m, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    X = 6 * np.random.rand(m, 1) - 4
    y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
    return X, y

def plot_learning_curves(model, X_train, y_train, X_test, y_test):
    train_errors = []
    test_errors = []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        test_errors.append(mean_squared_error(y_test_predict, y_test))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Train")
    plt.plot(np.sqrt(test_errors), "b-", linewidth=3, label="Test")
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Learning Curves")
    plt.show()

def create_polynomial_pipeline(degree):
    return Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])


X, y = generate_data(100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X_train, y_train, X_test, y_test)

degrees = [10, 2]
for degree in degrees:
    poly_pipeline = create_polynomial_pipeline(degree)
    plot_learning_curves(poly_pipeline, X_train, y_train, X_test, y_test)
