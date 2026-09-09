"""Unit tests for the model downloader.

huggingface_hub and the network are replaced with fakes via sys.modules,
so the suite runs with plain pytest - no torch, no downloads.
"""

import importlib
import sys
import types

import pytest

import download_models


@pytest.fixture()
def fake_hf(monkeypatch, tmp_path):
    module = types.ModuleType("huggingface_hub")
    calls = []

    def hf_hub_download(repo, name, local_dir):
        calls.append((repo, name, str(local_dir)))
        path = __import__("pathlib").Path(local_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * (2 * 1024 * 1024))
        return str(path)

    module.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setattr(download_models, "MODELS_DIR", tmp_path / "latam")
    monkeypatch.setattr(download_models, "CONDS_DIR", tmp_path / "conds")
    monkeypatch.setattr(download_models, "LATAM_FILES", [(name, 1_000) for name, _ in download_models.LATAM_FILES])
    monkeypatch.setattr(download_models, "BASE_FILES", [(name, 1_000) for name, _ in download_models.BASE_FILES])
    return calls


@pytest.fixture()
def fake_urlretrieve(monkeypatch, tmp_path):
    calls = []

    def urlretrieve(url, target):
        calls.append((url, str(target)))
        __import__("pathlib").Path(target).parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as fh:
            fh.write(b"\0" * (200 * 1024))

    monkeypatch.setattr("urllib.request.urlretrieve", urlretrieve)
    monkeypatch.setattr(download_models, "MODELS_DIR", tmp_path / "latam")
    monkeypatch.setattr(download_models, "CONDS_DIR", tmp_path / "conds")
    return calls


def test_fetch_skips_existing_large_file(fake_hf):
    target = download_models.MODELS_DIR / "t3_es_mx_latam.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\0" * (2 * 1024 * 1024))
    download_models.fetch("repo", "t3_es_mx_latam.safetensors", download_models.MODELS_DIR, 1_000_000)
    assert fake_hf == []


def test_fetch_downloads_missing_file(fake_hf):
    download_models.fetch("ResembleAI/X", "ve.pt", download_models.MODELS_DIR, 1_000_000)
    assert fake_hf == [("ResembleAI/X", "ve.pt", str(download_models.MODELS_DIR))]
    assert (download_models.MODELS_DIR / "ve.pt").is_file()


def test_fetch_raises_on_tiny_download(fake_hf, monkeypatch):
    module = sys.modules["huggingface_hub"]

    def tiny(repo, name, local_dir):
        path = __import__("pathlib").Path(local_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * 100)
        return str(path)

    monkeypatch.setattr(module, "hf_hub_download", tiny)
    with pytest.raises(RuntimeError):
        download_models.fetch("ResembleAI/X", "ve.pt", download_models.MODELS_DIR, 1_000_000)


def test_ensure_s3gen_pt_skips_existing(fake_hf, tmp_path):
    download_models.MODELS_DIR.mkdir()
    (download_models.MODELS_DIR / "s3gen.pt").write_bytes(b"existing")
    (download_models.MODELS_DIR / "s3gen_v3.pt").write_bytes(b"v3")
    download_models.ensure_s3gen_pt()
    assert (download_models.MODELS_DIR / "s3gen.pt").read_bytes() == b"existing"


def test_ensure_s3gen_pt_copies_v3(fake_hf):
    download_models.MODELS_DIR.mkdir()
    (download_models.MODELS_DIR / "s3gen_v3.pt").write_bytes(b"v3-weights")
    download_models.ensure_s3gen_pt()
    assert (download_models.MODELS_DIR / "s3gen.pt").read_bytes() == b"v3-weights"


def test_ensure_s3gen_pt_raises_when_missing(fake_hf):
    with pytest.raises(RuntimeError):
        download_models.ensure_s3gen_pt()


def test_ensure_default_voice_skips_existing(fake_urlretrieve):
    download_models.CONDS_DIR.mkdir()
    (download_models.CONDS_DIR / "default.wav").write_bytes(b"\0" * (200 * 1024))
    download_models.ensure_default_voice()
    assert fake_urlretrieve == []


def test_ensure_default_voice_downloads(fake_urlretrieve):
    download_models.ensure_default_voice()
    assert len(fake_urlretrieve) == 1
    assert (download_models.CONDS_DIR / "default.wav").is_file()


def test_main_end_to_end(fake_hf, fake_urlretrieve, capsys):
    assert download_models.main() == 0
    assert (download_models.MODELS_DIR / "t3_es_mx_latam.safetensors").is_file()
    assert (download_models.MODELS_DIR / "s3gen_v3.pt").is_file()
    assert (download_models.MODELS_DIR / "s3gen.pt").is_file()
    assert (download_models.MODELS_DIR / "grapheme_mtl_merged_expanded_v1.json").is_file()
    assert (download_models.MODELS_DIR / "ve.pt").is_file()
    assert (download_models.CONDS_DIR / "default.wav").is_file()
    assert "OK: checkpoints ready" in capsys.readouterr().out


def test_main_idempotent(fake_hf, fake_urlretrieve, capsys):
    download_models.main()
    out_first = capsys.readouterr().out
    download_models.main()
    out_second = capsys.readouterr().out
    assert "[skip]" in out_second
    assert "[get ]" not in out_second