import os
path = 'data/test_classification/medium/'
for filename in os.listdir(path):
    # e.g.: filename=='42.jpg' => new_name=='0042.jpg'
    new_name = '0'*(8-len(filename)) + filename
    os.rename(path+filename, path+new_name)