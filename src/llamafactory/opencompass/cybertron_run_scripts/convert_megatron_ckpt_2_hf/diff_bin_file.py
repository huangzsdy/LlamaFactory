import filecmp
import os
import sys

import torch


def torch_diff(file1, file2):
    t1 = torch.load(file1, map_location='cpu')
    t2 = torch.load(file2, map_location='cpu')
    keys1 = t1.keys()
    keys2 = t2.keys()
    if keys1 != keys2:
        print('The keys are different.')
    else:
        for key in keys1:
            if not torch.equal(t1[key], t2[key]):
                print(f'The key {key} is different.')
                return
    print('The files are same.')


def diff_bin_files(file1, file2):
    if filecmp.cmp(file1, file2, shallow=False):
        print('The files are same.')
    else:
        torch_diff(file1, file2)


if __name__ == '__main__':
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    assert os.path.exists(file1), f'{file1} does not exist.'
    assert os.path.exists(file2), f'{file2} does not exist.'
    diff_bin_files(file1, file2)
