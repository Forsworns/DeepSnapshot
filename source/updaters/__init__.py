import os
import importlib

def get_updater(name,denoiser,step_size):
    dir_name, _ = os.path.split(__file__)
    files = os.listdir(dir_name)
    for file in files:
        if file.endswith(".py"):
            net_name = file.rstrip('.py')
            if name == net_name:
                module_name = "updaters.{}".format(name) 
                module = importlib.import_module(module_name)
                class_name = name.title()
                cless = getattr(module,class_name)
                return cless(denoiser,step_size)
