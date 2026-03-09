import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from src.load_data import load_dataset
from src.preprocess import preprocess_data
from src.classical_model import train_classical_model
from src.quantum_kernel import quantum_kernel_matrix
from src.visualization import plot_kernel


def main():

    print("Loading dataset...")

    df = load_dataset(
        "data/train_transaction.csv",
        "data/train_identity.csv",
        sample_size=6000
    )

    print("Preprocessing data...")

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training classical model...")

    classical_model, classical_acc = train_classical_model(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("Classical Model Accuracy:", classical_acc)

    print("Computing quantum kernels...")

    K_train = quantum_kernel_matrix(X_train, X_train)
    K_test = quantum_kernel_matrix(X_test, X_train)

    print("Training quantum SVM...")

    qsvm = SVC(kernel="precomputed")

    qsvm.fit(K_train, y_train)

    preds = qsvm.predict(K_test)

    quantum_acc = accuracy_score(y_test, preds)

    print("Quantum Model Accuracy:", quantum_acc)

    print("Fraud Risk Scores:")

    risk_scores = qsvm.decision_function(K_test)

    print(risk_scores[:10])

    print("Generating kernel heatmap...")

    plot_kernel(K_train[:50, :50])


if __name__ == "__main__":
    main()
