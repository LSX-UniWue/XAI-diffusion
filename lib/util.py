import enum
import json

from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, Union, cast, get_args, get_origin

import tomli
import tomli_w
import zero

RawConfig = Dict[str, Any]
Report = Dict[str, Any]
T = TypeVar('T')


class TaskType(enum.Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

    def __str__(self) -> str:
        return self.value


class Timer(zero.Timer):
    @classmethod
    def launch(cls) -> 'Timer':
        timer = cls()
        timer.run()
        return timer


def _replace(data, condition, value):
    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)


_CONFIG_NONE = '__none__'


def unpack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x == _CONFIG_NONE, None))
    return config


def pack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x is None, _CONFIG_NONE))
    return config


def load_config(path: Union[Path, str]) -> Any:
    with open(path, 'rb') as f:
        return unpack_config(tomli.load(f))


def dump_config(config: Any, path: Union[Path, str]) -> None:
    with open(path, 'wb') as f:
        tomli_w.dump(pack_config(config), f)
    # check that there are no bugs in all these "pack/unpack" things
    assert config == load_config(path)


def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')


def from_dict(datacls: Type[T], data: dict) -> T:
    assert is_dataclass(datacls)
    data = deepcopy(data)
    for field in fields(datacls):
        if field.name not in data:
            continue
        if is_dataclass(field.type):
            data[field.name] = from_dict(field.type, data[field.name])
        elif (
                get_origin(field.type) is Union
                and len(get_args(field.type)) == 2
                and get_args(field.type)[1] is type(None)
                and is_dataclass(get_args(field.type)[0])
        ):
            if data[field.name] is not None:
                data[field.name] = from_dict(get_args(field.type)[0], data[field.name])
    return datacls(**data)
