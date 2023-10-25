from .trainer import Trainer
import imp
import os


def _wrapper_factory(cfg, network):
    module = '.'.join(['lib.train.trainers', cfg.task])
    path = os.path.join('lib/train/trainers', cfg.task+'.py')
    network_wrapper = imp.load_source(module, path).NetworkWrapper(network)
    return network_wrapper


def make_trainer(cfg, network):
    network = _wrapper_factory(cfg, network) ## 加载模型
    return Trainer(network)
