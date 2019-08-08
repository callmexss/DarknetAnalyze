import os
from functools import wraps
from typing import Union


def get_basename(path: Union[bytes, str, os.PathLike]):
    """Simple encapsulation of os.path.basename.

    :param path: a path
    :return: basename of given path
    """
    return os.path.basename(path)


def get_name_without_ext(path: Union[bytes, str, os.PathLike]):
    return os.path.splitext(get_basename(path))[0]


def get_ext_name(path: Union[bytes, str, os.PathLike]):
    """Get extension name of a file.

    :param path: a path of a file
    :return: extension name of a file if it has else return ""
    """
    return os.path.splitext(get_basename(path))[1]


def get_abs_path(path: Union[bytes, str, os.PathLike], *args):
    """Get absolute path of several parts.

    :param path: root path
    :param args: a list of parts to construct a path
    :return: generate absolute path with given arguments
    """
    return os.path.abspath(os.path.join(path, *args))


def join_name(path: Union[bytes, str, os.PathLike], *args):
    """Join path using given base path and several parts.

    :param path: root path
    :param args: a list of parts to construct a path
    :return: generate path
    """
    return os.path.join(path, *args)


def raise_exception_if_path_not_exists(func):
    def wrapper(path, *args, **kwargs):
        if not os.path.exists(path):
            raise OSError(f"{path} not exists.")
        func(path, *args, **kwargs)
    return wrapper
