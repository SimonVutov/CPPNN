"""Download dataset archives and verify published checksums before extraction."""
import argparse
import gzip
import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

MNIST = {
    "train-images-idx3-ubyte": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte": "ec29112dd5afa0611ce80d1b7f02629c",
}


def download(url, target, checksum):
    print(f"Downloading {url}", flush=True)
    digest = hashlib.md5()
    with urllib.request.urlopen(url, timeout=60) as source, target.open("wb") as output:
        while chunk := source.read(1024*1024):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != checksum:
        raise ValueError(f"Checksum mismatch for {url}")


def extract_cifar(archive_path, output):
    names = [f"data_batch_{i}.bin" for i in range(1, 6)]+["test_batch.bin", "batches.meta.txt"]
    with tarfile.open(archive_path, "r:gz") as archive:
        for name in names:
            member = archive.getmember("cifar-10-batches-bin/"+name)
            if not member.isfile() or (name.endswith(".bin") and member.size != 30730000):
                raise ValueError("Invalid CIFAR archive member")
            with archive.extractfile(member) as source, (output/name).open("wb") as target:
                shutil.copyfileobj(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["mnist", "cifar10"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output or root/("1mnist" if args.dataset == "mnist" else "2CIFAR10/cifar-10-batches-bin")
    output.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary)/"download"
            if args.dataset == "mnist":
                for name, checksum in MNIST.items():
                    download("https://ossci-datasets.s3.amazonaws.com/mnist/"+name+".gz", archive, checksum)
                    with gzip.open(archive, "rb") as source, (output/name).open("wb") as target:
                        shutil.copyfileobj(source, target)
            else:
                download("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", archive,
                         "c32a1d4ab5d03f1284b67883e8d87530")
                extract_cifar(archive, output)
        print(output)
    except (OSError, ValueError, tarfile.TarError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
