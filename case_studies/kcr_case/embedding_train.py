import pandas as pd
pd.options.mode.chained_assignment = None
from word_embedding_main import WordEmbedding
import numpy as np

def get_fasta_seq(file):
    infile = open(file)
    spe_character = []
    sequences = []
    for line in infile:
        if (line.startswith('>') == False) and (any(x in line for x in spe_character) == False):
                sequence = ''.join(line).replace('\n', '')
                sequences.append(sequence)

    infile.close()
    return sequences


def main():
    sequences_pos = np.array(pd.read_csv('data/train/pos_train.txt'))
    sequences_neg =  np.array(pd.read_csv('data/train/neg_train.txt'))
    sequences = np.append(sequences_pos,sequences_neg)
    w2v = WordEmbedding(w2v=1, sg=1, ngram_len=1, vectordim= 5)
    seqs_tokenized = w2v.tokenize_sequences(sequences)
    w2v.train_wordembedding(seqs_tokenized=seqs_tokenized, filename= 'w2v_sg_20_5')

if __name__ == "__main__":
    main()