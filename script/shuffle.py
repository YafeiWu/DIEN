import os
import sys
import random
import time

import tempfile
from subprocess import call


def main(file, temporary=False):
    start_ = time.time()
    tf_os, tpath = tempfile.mkstemp(dir='/tmp')
    tf = open(tpath, 'w')

    fd = open(file, "r")
    for l in fd:
        print >> tf, l.strip("\n")
    tf.close()

    lines = open(tpath, 'r').readlines()
    random.shuffle(lines)
    if temporary:
        path, filename = os.path.split(os.path.realpath(file))
        fd = tempfile.TemporaryFile(prefix=filename + '.shuf', dir=path)
    else:
        fd = open(file + '.shuf', 'w')

    for l in lines:
        s = l.strip("\n")
        print >> fd, s

    if temporary:
        fd.seek(0)
    else:
        fd.close()

    os.remove(tpath)

    time_used = time.time() - start_
    print("shuffle time used : {} \t file name : {}".format(time_used, file))

    return fd

def shuffle_file(file, temporary=False):
    start_ = time.time()
    tf_os, tpath = tempfile.mkstemp(dir='/tmp')
    tf = open(tpath, 'w')

    fd = open(file, "r")
    for l in fd:
        print >> tf, l.strip("\n")
    tf.close()

    lines = open(tpath, 'r').readlines()
    random.shuffle(lines)
    if temporary:
        path, filename = os.path.split(os.path.realpath(file))
        fd = tempfile.TemporaryFile(prefix=filename + '.shuf', dir=path)
    else:
        shuffled_filename = file + '.shuf'
        fd = open(shuffled_filename, 'w')

    for l in lines:
        s = l.strip("\n")
        print >> fd, s

    if temporary:
        fd.seek(0)
    else:
        fd.close()

    os.remove(tpath)

    time_used = time.time() - start_
    print("shuffle time used : {} \t file name : {}".format(time_used, file))

    return shuffled_filename


if __name__ == '__main__':
    main(sys.argv[1])

