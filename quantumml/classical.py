from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss


def train_classical(X_train, y_train, X_test, y_test, seed=21):
    model = MLPClassifier(hidden_layer_sizes=(8, 8), activation="tanh", solver="lbfgs",
                          alpha=0.01, max_iter=700, random_state=seed)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return model, {"accuracy": accuracy_score(y_test, model.predict(X_test)),
                   "loss": log_loss(y_test, probs)}, probs
