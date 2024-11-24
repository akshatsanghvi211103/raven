import os
print(os.path.abspath(__file__))
dirpath = os.path.dirname(os.path.abspath(__file__))
print(dirpath)

model_path = os.path.join(dirpath, "spm", "unigram", "unigram1000.model")
print(model_path, os.path.exists(model_path))
