from src.load_data import load_dataset
from src.preprocess import preprocess_data
from src.visualization import plot_kernel

from src.experiment_runner import run_experiment
from src.metrics import evaluate_model
from src.fraud_scorer import fraud_probability, print_top_risky_transactions
from src.quantum_analysis import kernel_statistics, kernel_sparsity


def main():

    print("Loading Dataset")

    df = load_dataset(
        "data/train_transaction.csv",
        "data/train_identity.csv",
        sample_size=6000
    )

    print("Preprocessing")

    X, y = preprocess_data(df)

    preds, scores, y_test, K = run_experiment(X, y)

    evaluate_model(y_test, preds, scores)

    probs = fraud_probability(scores)

    print_top_risky_transactions(probs)

    kernel_statistics(K)

    kernel_sparsity(K)

    plot_kernel(K[:50, :50])


if __name__ == "__main__":
    main()
