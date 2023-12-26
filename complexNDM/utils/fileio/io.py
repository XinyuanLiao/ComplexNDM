import datetime
import os
from pathlib import Path

from .misc import is_str, is_list_of
from .handlers import BaseFileHandler, JsonHandler, PickleHandler, YamlHandler


file_handlers = {
    "json": JsonHandler(),
    "yaml": YamlHandler(),
    "yml": YamlHandler(),
    "pickle": PickleHandler(),
    "pkl": PickleHandler(),
}


def load(file, file_format=None, **kwargs):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and is_str(file):
        file_format = file.split(".")[-1]
    if file_format not in file_handlers:
        raise TypeError("Unsupported format: {}".format(file_format))

    handler = file_handlers[file_format]
    if is_str(file):
        obj = handler.load_from_path(file, **kwargs)
    elif hasattr(file, "read"):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def find_new_file_with_ext(dir, ext='pt'):

    list = os.listdir(dir)
    list.sort(key=lambda fn:os.path.getmtime(dir+'/'+fn))
    list.reverse()

    file = None
    for i in range(len(list)):
        if list[i].split('.')[-1] == ext:
            file = list[i]
            break
    if file != None:
        file = os.path.join(dir, file)
        
    return file
