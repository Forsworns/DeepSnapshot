import importlib
import os


def get_updater(u_name, denoiser, step_size):
    dir_name, _ = os.path.split(__file__)
    files = os.listdir(dir_name)
    for file in files:
        if file.endswith(".py"):
            file_name = file[:-3]
            if u_name == file_name:
                module_name = "updaters.{}".format(u_name)
                module = importlib.import_module(module_name)
                class_name = u_name.title()
                cless = getattr(module, class_name)
                return cless(denoiser, step_size)
