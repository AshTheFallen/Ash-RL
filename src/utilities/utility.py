import os

# creating a directory if it does not exist - utility function
def maybe_make_dir(dir):
    if not os.path.exists(dir):
        print('creating directory')
        os.mkdir(dir)