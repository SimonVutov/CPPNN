import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import invariant
from model import MLP


def read_batch(path):
    raw = np.fromfile(path, dtype=np.uint8)
    if not raw.size or raw.size % 3073:
        raise ValueError(f"Invalid CIFAR binary batch: {path}")
    records = raw.reshape(-1, 3073)
    if np.any(records[:, 0] >= 10):
        raise ValueError(f"Invalid CIFAR label: {path}")
    images = records[:, 1:].reshape(-1, 3, 16, 2, 16, 2).astype(np.float32)
    features = (images.mean(axis=(3, 5))/255).reshape(-1, 768)
    return features, records[:, 0].astype(np.int64)


def load_data(directory, training=True):
    paths = [directory/f"data_batch_{i}.bin" for i in range(1, 6)] if training else [directory/"test_batch.bin"]
    batches = [read_batch(path) for path in paths]
    return np.concatenate([x for x, _ in batches]), np.concatenate([y for _, y in batches])


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 MLP with Invariant C++ matrix products")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1]/"2CIFAR10/cifar-10-batches-bin")
    parser.add_argument("--output", type=Path, default=Path("runs/invariant"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--evaluate", type=Path, help="Evaluate a saved checkpoint without training")
    args = parser.parse_args()
    if min(args.epochs, args.hidden, args.batch_size) <= 0 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("sizes, epochs, and learning rate must be positive")
    if args.seed < 0 or any(v is not None and v <= 0 for v in (args.train_limit, args.test_limit)):
        parser.error("seed must be nonnegative and limits positive")
    try:
        test_x, test_y = load_data(args.data, False)
        test_x, test_y = test_x[:args.test_limit], test_y[:args.test_limit]
        if args.evaluate:
            model, mean, scale = MLP.load(args.evaluate)
            print(json.dumps({"test_accuracy": model.accuracy((test_x-mean)/scale, test_y), "test_samples": len(test_y)}))
            return
        x, y = load_data(args.data)
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(x))[:args.train_limit]
        x, y = x[order], y[order]
        if len(x) < 20:
            parser.error("Need at least 20 training examples for a validation split")
        validation_count = max(1, len(x)//10)
        val_x, val_y = x[:validation_count], y[:validation_count]
        x, y = x[validation_count:], y[validation_count:]
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), 1e-4)
        x, val_x, test_x = [(data-mean)/scale for data in (x, val_x, test_x)]
        model = MLP(x.shape[1], args.hidden, seed=args.seed)
        args.output.mkdir(parents=True, exist_ok=True)
        history, best = [], -1
        start = time.perf_counter()
        for epoch in range(args.epochs):
            epoch_start = time.perf_counter()
            order = rng.permutation(len(x))
            loss = 0
            rate = args.learning_rate * (0.2 if epoch >= args.epochs*0.75 else 1)
            for offset in range(0, len(x), args.batch_size):
                indices = order[offset:offset+args.batch_size]
                batch_loss, gradients = model.loss_gradients(x[indices], y[indices], weight_decay=1e-4)
                if not np.isfinite(batch_loss):
                    raise ValueError("Training diverged; lower the learning rate")
                model.step(gradients, rate)
                loss += batch_loss*len(indices)
            accuracy = model.accuracy(val_x, val_y)
            seconds = time.perf_counter()-epoch_start
            row = {"epoch": epoch+1, "loss": loss/len(x), "validation_accuracy": accuracy,
                   "seconds": seconds, "samples_per_second": len(x)/seconds}
            history.append(row)
            print(json.dumps(row), flush=True)
            if accuracy > best:
                best = accuracy
                model.save(args.output/"best.npz", mean, scale)
        model, _, _ = MLP.load(args.output/"best.npz")
        evaluation_start = time.perf_counter()
        test_accuracy = model.accuracy(test_x, test_y)
        report = {"test_accuracy": test_accuracy, "best_validation_accuracy": best,
                  "test_seconds": time.perf_counter()-evaluation_start,
                  "total_seconds": time.perf_counter()-start, "train_samples": len(x),
                  "validation_samples": len(val_y), "test_samples": len(test_y), "history": history,
                  "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "platform": platform.platform(), "numpy": np.__version__, "invariant": invariant.__version__}
        (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({k: report[k] for k in ("test_accuracy", "best_validation_accuracy", "total_seconds")}))
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
