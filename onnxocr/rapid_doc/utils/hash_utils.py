# Copyright (c) Opendatalab. All rights reserved.
import hashlib
import json
from enum import Enum


def bytes_md5(file_bytes):
    hasher = hashlib.md5()
    hasher.update(file_bytes)
    return hasher.hexdigest().upper()


def str_md5(input_string):
    hasher = hashlib.md5()
    # 在Python3中，需要将字符串转化为字节对象才能被哈希函数处理
    input_bytes = input_string.encode('utf-8')
    hasher.update(input_bytes)
    return hasher.hexdigest()


def str_sha256(input_string):
    hasher = hashlib.sha256()
    # 在Python3中，需要将字符串转化为字节对象才能被哈希函数处理
    input_bytes = input_string.encode('utf-8')
    hasher.update(input_bytes)
    return hasher.hexdigest()


def dict_md5(d):
    json_str = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(json_str.encode('utf-8')).hexdigest()

def make_hashable(value):
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if k == "custom_model":
                result[k] = type(v).__name__
            else:
                result[k] = make_hashable(v)
        return json.dumps(result, sort_keys=True)
    elif isinstance(value, list):
        return json.dumps([make_hashable(v) for v in value], sort_keys=True)
    elif isinstance(value, Enum):
        return value.name  # 或 value.value
    return value