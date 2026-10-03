"""Two-layer classifier; all dense products use Invariant's C++ backend."""
import numpy as np
from invariant import matmul


class MLP:
    def __init__(self, inputs, hidden, classes=10, seed=42):
        if min(inputs, hidden, classes) <= 0:
            raise ValueError("Layer sizes must be positive")
        rng = np.random.default_rng(seed)
        self.params = {
            "w1": (rng.standard_normal((inputs, hidden))*np.sqrt(2/inputs)).astype(np.float32),
            "b1": np.zeros(hidden, np.float32),
            "w2": (rng.standard_normal((hidden, classes))*np.sqrt(2/hidden)).astype(np.float32),
            "b2": np.zeros(classes, np.float32),
        }
        self.velocity = {k: np.zeros_like(v) for k, v in self.params.items()}

    def logits(self, x):
        hidden = np.maximum(matmul(x, self.params["w1"])+self.params["b1"], 0)
        return matmul(hidden, self.params["w2"])+self.params["b2"]

    def loss_gradients(self, x, labels, weight_decay=0):
        if x.ndim != 2 or len(x) == 0 or labels.shape != (len(x),):
            raise ValueError("Expected a nonempty batch and one label per row")
        if labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels >= len(self.params["b2"])):
            raise ValueError("Invalid class labels")
        p = self.params
        hidden = np.maximum(matmul(x, p["w1"])+p["b1"], 0)
        logits = matmul(hidden, p["w2"])+p["b2"]
        shifted = logits-logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        sums = exp.sum(axis=1, keepdims=True)
        loss = np.mean(np.log(sums[:, 0])-shifted[np.arange(len(x)), labels])
        loss += weight_decay/2 * (np.sum(p["w1"]**2)+np.sum(p["w2"]**2))
        delta = exp/sums
        delta[np.arange(len(x)), labels] -= 1
        delta /= len(x)
        dh = matmul(delta, p["w2"].T)*(hidden > 0)
        grads = {"w2": matmul(hidden.T, delta)+weight_decay*p["w2"],
                 "b2": delta.sum(axis=0),
                 "w1": matmul(x.T, dh)+weight_decay*p["w1"],
                 "b1": dh.sum(axis=0)}
        return float(loss), grads

    def step(self, gradients, learning_rate, momentum=0.9):
        for key in self.params:
            self.velocity[key] *= momentum
            self.velocity[key] -= learning_rate*gradients[key]
            self.params[key] += self.velocity[key]

    def accuracy(self, x, y, batch_size=256):
        if len(x) == 0 or len(x) != len(y) or batch_size <= 0:
            raise ValueError("Invalid evaluation batch")
        correct = 0
        for start in range(0, len(x), batch_size):
            predictions = self.logits(x[start:start+batch_size]).argmax(axis=1)
            correct += np.count_nonzero(predictions == y[start:start+batch_size])
        return float(correct/len(x))

    def save(self, path, mean, scale):
        np.savez_compressed(path, **self.params, mean=mean, scale=scale, format_version=np.array(1))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as state:
            if state["format_version"].item() != 1:
                raise ValueError("Unsupported checkpoint version")
            w1, w2 = state["w1"], state["w2"]
            if w1.ndim != 2 or w2.ndim != 2 or w1.shape[1] != w2.shape[0]:
                raise ValueError("Invalid checkpoint shapes")
            model = cls(w1.shape[0], w1.shape[1], w2.shape[1])
            for key in model.params:
                value = state[key]
                if value.shape != model.params[key].shape or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError("Invalid checkpoint parameter")
                model.params[key] = value.copy()
            mean, scale = state["mean"], state["scale"]
            if mean.shape != (w1.shape[0],) or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("Invalid checkpoint normalization")
            return model, mean.copy(), scale.copy()
