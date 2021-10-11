#encoding:utf-8
from .BaseConfig import BaseConfig
class GloveConfig(BaseConfig):
    model = 'Glove'
    freeze = True
    avg=''