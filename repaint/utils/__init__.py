import yaml
import os

def txtread(path):
    path = os.path.expanduser(path)
    with open(path, 'r') as f:
        return f.read()


def yamlread(path):
    return yaml.safe_load(txtread(path=path))
