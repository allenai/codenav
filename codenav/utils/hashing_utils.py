import hashlib


def md5_hash_str(to_hash: str) -> str:
    return hashlib.md5(to_hash.encode()).hexdigest()


def md5_hash_file(to_hash_path: str):
    with open(to_hash_path, "r") as f:
        return md5_hash_str(f.read())
