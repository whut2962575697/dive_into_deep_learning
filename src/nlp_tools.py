import torch
import numpy as np
from tqdm import tqdm 


def generate_dataset(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as f:
        corpus_chars  = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')[:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_inx = {char:idx for idx, char in enumerate(idx_to_char)}
    vocab_size = len(idx_to_char)
    corpus_words = [char_to_inx[char] for char in corpus_chars]
    return corpus_chars, corpus_words, idx_to_char, char_to_inx, vocab_size

def data_iter_random(corpus_words, batch_size, sqe_len, shuffle=True):

    sample_nums = (len(corpus_words)-1) // sqe_len
    sample_start_indexs = [i*sqe_len for i in range(sample_nums)]
    if shuffle:
        np.random.shuffle(sample_start_indexs)
    for i in range(0, sample_nums, batch_size):
        batch_sample_start_indexs = sample_start_indexs[i:i+batch_size]

        batch_x = list()
        batch_y = list()
        for batch_sample_start_index in batch_sample_start_indexs:
            batch_x.append(corpus_words[batch_sample_start_index:batch_sample_start_index+sqe_len])
            batch_y.append(corpus_words[batch_sample_start_index+1:batch_sample_start_index+sqe_len+1])

        yield torch.Tensor(batch_x), torch.Tensor(batch_y)

def data_iter_consecutive(corpus_words, batch_size, sqe_len):
    corpus_words_max_len = len(corpus_words) // batch_size * batch_size
    corpus_words = corpus_words[:corpus_words_max_len]
    corpus_words = torch.Tensor(corpus_words)
    corpus_words = corpus_words.view(batch_size, -1)
    batch_nums = (corpus_words.shape[1]-1) // sqe_len
    for i in range(batch_nums):
        batch_x = corpus_words[:, i*sqe_len:(i+1)*sqe_len]
        batch_y = corpus_words[:, i*sqe_len+1:(i+1)*sqe_len+1]
        yield batch_x, batch_y






if __name__ == "__main__":
    corpus_chars, corpus_words, idx_to_char, char_to_inx, vocab_size = generate_dataset(r'G:\xin.src\dive_into_deep_learning\dataset\jaychou_lyrics/jaychou_lyrics.txt')

    data_loder = data_iter_consecutive(corpus_words, 2, 6)
    for x, y in tqdm(data_loder, leave=False):
        print(x, y)


    



    
