import dataclasses
import io
from typing import Optional

import dacite
import yaml


@dataclasses.dataclass()
class DatabaseConfiguration:
    db: str
    uname: str
    pw: str
    host: str
    port: int
    fuzzy_dbs: Optional[dict]
    huggingface: Optional[str]

def load_db_configuration(file: str | io.IOBase, data_class=DatabaseConfiguration):
    f = file
    arg = None
    if isinstance(file, str):
        f = open(file, "r")
    if isinstance(f, io.IOBase):
        arg = yaml.safe_load(f)
    if isinstance(arg, dict):
        arg = dacite.from_dict(data_class=data_class, data=arg)
    f.close()
    return arg


def load_list_configuration(file: str | io.IOBase, data_class=DatabaseConfiguration):
    f = file
    arg = None
    if isinstance(file, str):
        f = open(file, "r")
    if isinstance(f, io.IOBase):
        arg = yaml.safe_load(f)
    if isinstance(arg, list):
        arg = [dacite.from_dict(data_class=data_class, data=x) for x in arg]
    f.close()
    return arg
