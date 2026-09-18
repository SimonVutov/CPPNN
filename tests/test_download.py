import hashlib
import io
import sys
import tarfile
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import download_data


def test_checksum(tmp_path, monkeypatch):
    payload = b"example dataset bytes"
    monkeypatch.setattr(download_data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    target = tmp_path/"download"
    download_data.download("https://example.invalid", target, hashlib.md5(payload).hexdigest())
    assert target.read_bytes() == payload
    with pytest.raises(ValueError, match="Checksum"):
        download_data.download("https://example.invalid", target, "wrong")


def test_archive_link_rejected(tmp_path):
    path = tmp_path/"archive.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("cifar-10-batches-bin/data_batch_1.bin")
        member.type = tarfile.SYMTYPE
        member.linkname = "outside"
        archive.addfile(member)
    with pytest.raises(ValueError, match="Invalid CIFAR"):
        download_data.extract_cifar(path, tmp_path)
