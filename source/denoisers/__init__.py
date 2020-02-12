import importlib
import os


def get_denoiser(d_name, channels, **kwargs):
    dir_name, _ = os.path.split(__file__)
    files = os.listdir(dir_name)
    for file in files:
        if file.endswith("net.py"):
            file_name = file[:-3]
            if d_name == file_name:
                module_name = "denoisers.{}".format(d_name)
                module = importlib.import_module(module_name)
                class_name = "{}Net".format(d_name.title()[:-3])
                cless = getattr(module, class_name)
                return cless(channels)
