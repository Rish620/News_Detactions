from news_detector.training import train_and_save


def main() -> None:
    metrics = train_and_save()
    print("Model training complete!")
    print(f"Total samples: {metrics['samples']}")
    print(f"Train samples: {metrics['train_samples']}")
    print(f"Test samples: {metrics['test_samples']}")
    print(f"Test accuracy: {metrics['test_accuracy']:.2%}")
    if metrics["cross_validation_accuracy_mean"] is not None:
        print(
            "Cross-validation accuracy: "
            f"{metrics['cross_validation_accuracy_mean']:.2%} "
            f"+/- {metrics['cross_validation_accuracy_std']:.2%}"
        )


if __name__ == "__main__":
    main()
