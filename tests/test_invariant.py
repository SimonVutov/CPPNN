import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"3CIFAR10-invariant"))
from model import MLP
from train import read_batch


def test_numpy_parity_and_gradients():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, 3)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 1])
    model = MLP(3, 4, 2, seed=2)
    p = model.params
    reference = np.maximum(x @ p["w1"]+p["b1"], 0) @ p["w2"]+p["b2"]
    np.testing.assert_allclose(model.logits(x), reference, atol=1e-6)
    _, gradients = model.loss_gradients(x, y, 0.01)
    for key, param in p.items():
        for index in np.ndindex(param.shape):
            original = param[index].copy()
            param[index] = original+1e-3
            plus = model.loss_gradients(x, y, 0.01)[0]
            param[index] = original-1e-3
            minus = model.loss_gradients(x, y, 0.01)[0]
            param[index] = original
            np.testing.assert_allclose(gradients[key][index], (plus-minus)/0.002, atol=2e-4, rtol=0.02)


def test_overfit_and_roundtrip(tmp_path):
    x = np.array([[-1,-1], [-1,1], [1,-1], [1,1]], np.float32)
    y = np.array([0,1,1,0])
    model = MLP(2, 16, 2)
    initial = model.loss_gradients(x, y)[0]
    for _ in range(150):
        loss, grads = model.loss_gradients(x, y)
        model.step(grads, 0.05)
    assert loss < initial*0.05
    assert model.accuracy(x,y) == 1
    path = tmp_path/"model.npz"
    model.save(path, np.zeros(2, np.float32), np.ones(2, np.float32))
    restored, mean, scale = MLP.load(path)
    np.testing.assert_array_equal(model.logits(x), restored.logits((x-mean)/scale))


def test_binary_reader(tmp_path):
    path = tmp_path/"batch.bin"
    records = np.zeros((2,3073), np.uint8)
    records[:,0] = [1,9]
    records[1,1:] = 255
    records.tofile(path)
    x,y = read_batch(path)
    assert x.shape == (2,768)
    np.testing.assert_array_equal(y,[1,9])
    assert np.all(x[0]==0) and np.all(x[1]==1)
    path.write_bytes(b"bad")
    with pytest.raises(ValueError): read_batch(path)
    records[0,0]=10
    records.tofile(path)
    with pytest.raises(ValueError): read_batch(path)


def test_training_cli(tmp_path):
    import json
    import subprocess
    directory = tmp_path/"data"
    directory.mkdir()
    rng = np.random.default_rng(3)
    for name in [f"data_batch_{i}.bin" for i in range(1,6)]+["test_batch.bin"]:
        records = rng.integers(0,256,(20,3073),dtype=np.uint8)
        records[:,0] %= 10
        records.tofile(directory/name)
    script = Path(__file__).resolve().parents[1]/"3CIFAR10-invariant/train.py"
    output = tmp_path/"output"
    command = [sys.executable, str(script), "--data", str(directory)]
    subprocess.run(command+["--output", str(output), "--epochs", "2", "--hidden", "8"], check=True, capture_output=True, text=True)
    metrics = json.loads((output/"metrics.json").read_text())
    assert metrics["train_samples"] == 90
    assert metrics["validation_samples"] == 10
    restored = subprocess.run(command+["--evaluate", str(output/"best.npz")], check=True, capture_output=True, text=True)
    assert json.loads(restored.stdout)["test_accuracy"] == metrics["test_accuracy"]
    bad = subprocess.run(command+["--epochs", "0"], capture_output=True, text=True)
    assert bad.returncode != 0 and "must be positive" in bad.stderr
