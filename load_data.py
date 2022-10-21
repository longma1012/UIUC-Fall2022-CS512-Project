print("------load_data.py------")
import os
import gzip

def read_gz_file(path):
    f = gzip.open(path, 'rb')
    for line in f.readline():
        print(line)
    return

path = 'SNAP Dataset ego-facebook/facebook.tar.gz'
read_gz_file(path)