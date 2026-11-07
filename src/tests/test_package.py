import importlib


def test_package_imports() -> None:
    module = importlib.import_module("getwork.preprocess")
    assert module.__name__ == "getwork.preprocess"
