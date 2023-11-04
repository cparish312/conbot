""""Script for collecting files for Training.""" 

import os
import subprocess
import filecmp


def get_files(root_dir, ext, save_dir):
    save_files = set()
    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            f = os.path.join(path, name)
            if name.split('.')[-1] == ext:
                save_files.add(f)

    copy_count = 0
    for f in save_files:
        fname = f.split('/')[-1]
        f_dest = os.path.join(save_dir, fname)
        if os.path.exists(f_dest):
            if not filecmp.cmp(f, f_dest): # check that new file
                f_base = fname.split('.')[0]
                f_ext = '.'.join(fname.split('.')[1:])
                fname = f_base + '1' + f_ext
                f_dest = os.path.join(save_dir, fname)
                subprocess.call(['cp' , f, f_dest])
                copy_count += 1
        else:
            subprocess.call(['cp' , f, f_dest])
            copy_count += 1
    print(f'Total Files: {len(save_files)}')
    print(f'New Files: {copy_count}')


if __name__ == '__main__':
    root_dir = '/Users/connorparish/dzd_repos/'
    ext = 'py'
    save_dir = '../data/python_files'

    get_files(root_dir, ext, save_dir)

        

    
    

