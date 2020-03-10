import json


class PoseConfigLoader(object):
    def __init__(self, filepath):
        with open(filepath, 'r') as file:
            self.__dict__ = json.loads(file.read())
