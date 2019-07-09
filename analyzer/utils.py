import os
from functools import wraps


def get_basename(path):
    return os.path.basename(path)


def get_name_without_ext(path):
    return os.path.splitext(get_basename(path))[0]


def get_ext_name(path):
    return os.path.splitext(get_basename(path))[1]


def raise_exception_if_path_not_exists(func):
    def wrapper(path, *args, **kwargs):
        if not os.path.exists(path):
            raise OSError(f"{path} not exists.")
        func(path, *args, **kwargs)
    return wrapper
