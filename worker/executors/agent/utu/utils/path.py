import hashlib
import pathlib
import tempfile
from typing import Any
from urllib.parse import urlparse

import requests
import yaml


def get_package_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.parent


DIR_ROOT = get_package_path()
CACHE_DIR = DIR_ROOT / ".cache"


class FileUtils:
    @staticmethod
    def is_web_url(url: str) -> bool:
        parsed_url = urlparse(url)
        return all([parsed_url.scheme, parsed_url.netloc])

    @staticmethod
    def get_file_ext(file_path: str) -> str:
        if FileUtils.is_web_url(file_path):
            return pathlib.Path(urlparse(file_path).path).suffix
        return pathlib.Path(file_path).suffix

    @staticmethod
    def download_file(url: str, save_path: str = None) -> str:
        """Download file from web. Return the saved path"""
        # if not save_path, use tempfile
        if not save_path:
            save_path = tempfile.NamedTemporaryFile(
                suffix=FileUtils.get_file_ext(url),
                delete=False,
            ).name
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path

    @staticmethod
    def get_file_md5(file_path: str) -> str:
        """Clac md5 for local or web file"""
        hash_md5 = hashlib.md5()
        if FileUtils.is_web_url(file_path):
            with requests.get(file_path, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=4096):
                    hash_md5.update(chunk)
        else:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def load_yaml(file_path: pathlib.Path | str) -> dict[str, Any]:
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        with file_path.open() as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_prompts(fn: str | pathlib.Path) -> dict[str, str]:
        """Load prompts from yaml file.

        - Default path: `DIR_ROOT / "utu/prompts" / fn`
        """
        if isinstance(fn, str):
            if not fn.endswith(".yaml"):
                fn += ".yaml"
            fn = DIR_ROOT / "utu" / "prompts" / fn
        assert fn.exists(), f"File {fn} does not exist!"
        with fn.open(encoding="utf-8") as f:
            return yaml.safe_load(f)
