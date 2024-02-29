import os

def make_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d)
