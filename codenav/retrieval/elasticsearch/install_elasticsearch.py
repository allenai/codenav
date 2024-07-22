import hashlib
import os
import shutil
import sys
import urllib
import urllib.request
from pathlib import Path

ABS_PATH_OF_ES_DIR = os.path.abspath(os.path.dirname(Path(__file__)))

DOWNLOAD_DIR = os.path.join(os.path.expanduser("~/.cache/codenav/elasticsearch"))

ES_VERSION = "8.12.1"
KIBANA_VERSION = ES_VERSION

ES_PATH = os.path.join(DOWNLOAD_DIR, f"elasticsearch-{ES_VERSION}")
KIBANA_PATH = os.path.join(DOWNLOAD_DIR, f"kibana-{ES_VERSION}")

PLATFORM_TO_ES_URL = {
    "linux-x86_64": f"https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-{ES_VERSION}-linux-x86_64.tar.gz",
    "darwin-aarch64": f"https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-{ES_VERSION}-darwin-aarch64.tar.gz",
    "darwin-x86_64": f"https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-{ES_VERSION}-darwin-x86_64.tar.gz",
}

PLATFORM_TO_KIBANA_URL = {
    "linux-x86_64": f"https://artifacts.elastic.co/downloads/kibana/kibana-{KIBANA_VERSION}-linux-x86_64.tar.gz",
    "darwin-aarch64": f"https://artifacts.elastic.co/downloads/kibana/kibana-{KIBANA_VERSION}-darwin-aarch64.tar.gz",
    "darwin-x86_64": f"https://artifacts.elastic.co/downloads/kibana/kibana-{KIBANA_VERSION}-darwin-x86_64.tar.gz",
}


def compute_sha512(file_path: str):
    sha512_hash = hashlib.sha512()
    with open(file_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha512_hash.update(byte_block)
    return sha512_hash.hexdigest()


def install_from_url(url: str) -> None:
    print(f"Downloading {url}")

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    name = "-".join(url.split("/")[-1].split("-")[:2])

    if os.path.exists(os.path.join(DOWNLOAD_DIR, name)):
        print(f"{name} already exists. Skipping download...")
        return

    tar_path = os.path.join(DOWNLOAD_DIR, f"{name}.tar.gz")
    tar_hash_path = tar_path + ".sha512"

    urllib.request.urlretrieve(url, tar_path)
    urllib.request.urlretrieve(
        url + ".sha512",
        tar_hash_path,
    )
    print("Download complete")

    # Checking SHA512
    print("Checking SHA512")
    sha512 = compute_sha512(tar_path)
    with open(tar_hash_path, "r") as f:
        expected_sha512 = f.read().strip().split(" ")[0]

    if sha512 != expected_sha512:
        raise ValueError(f"SHA512 mismatch. Expected {expected_sha512}, got {sha512}")

    print(f"Extracting {tar_path} to {DOWNLOAD_DIR}")
    os.system(f"tar -xzf {tar_path} -C {DOWNLOAD_DIR}")

    assert os.path.join(ES_PATH)

    os.remove(tar_path)
    os.remove(tar_hash_path)

    print(f"{name} installation complete")


def install_elasticsearch():
    if sys.platform == "darwin":
        if os.uname().machine == "arm64":
            platform = "darwin-aarch64"
        else:
            platform = "darwin-x86_64"
    else:
        assert sys.platform == "linux"
        platform = "linux-x86_64"

    install_from_url(PLATFORM_TO_ES_URL[platform])

    # Copy the elasticsearch config file to the elasticsearch directory
    es_config_file = os.path.join(ABS_PATH_OF_ES_DIR, "elasticsearch.yml")
    shutil.copy(es_config_file, os.path.join(ES_PATH, "config"))

    install_from_url(PLATFORM_TO_KIBANA_URL[platform])


def is_es_installed():
    return os.path.exists(ES_PATH) and os.path.exists(KIBANA_PATH)


if __name__ == "__main__":
    if not is_es_installed():
        install_elasticsearch()
    else:
        print(f"Elasticsearch already installed at {ES_PATH}")
