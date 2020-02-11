import os
import importlib

def get_denoiser(name, channels, **kwargs):
    dir_name, _ = os.path.split(__file__)
    files = os.listdir(dir_name)
    for file in files:
        if file.endswith("net.py"):
            net_name = file.rstrip('.py')
            if name == net_name:
                module_name = "denoisers.{}".format(name) 
                module = importlib.import_module(module_name)
                class_name = "{}Net".format(name.title()[:-3])
                cless = getattr(module,class_name)
                return cless(channels)
