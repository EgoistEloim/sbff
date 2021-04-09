import numpy as np
from IPython import embed
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec



def read_glove_filter(glove_path, needed_set):
    embedding = {}
    with open(glove_path, 'r') as f:
        for line in f:
            line = line.split()
            word = line[0]
            if word in needed_set:
                vector = line[1:]
                embedding[word] = vector
    return embedding

def save_kv(path, tmp_path, save_path):
    glove_file = datapath(path)
    tmp_file = get_tmpfile(tmp_path)
    _ = glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    model.save(save_path)
    print("Save finished.")
    return model


def save_mini_glove(embedding, save_mini_glove_path):
    try:
        with open(save_mini_glove_path, 'w') as wf:
            for each in embedding.keys():
                tmp = each + ' ' + ' '.join(embedding[each]) + '\n'
                wf.write(tmp)
        return True
    except:
        return False

if __name__ == '__main__':
    path = '/Users/chenhang/Downloads/glove/glove.6B.50d.txt'
    tmp_path = 'test_word2vec.txt'
    save_path = 'sbff.kv'
    check = set()
    check.add('for')
    check.add('.')
    save_mini_glove_path = '/Users/chenhang/PycharmProjects/0510-1/tttttmp.txt'
    embedding = read_glove_filter(glove_path=path, needed_set=check)
    save_flag = save_mini_glove(embedding=embedding, save_mini_glove_path=save_mini_glove_path)
    save_kv(path=save_mini_glove_path, tmp_path=tmp_path, save_path=save_path)


