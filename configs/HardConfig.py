#encoding:utf-8
from .BaseConfig import BaseConfig
class HardConfig(BaseConfig):
    model = 'Hard'
    freeze = True
    avg=''