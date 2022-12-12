import pandas as pd
pd.options.mode.chained_assignment = None
#from WordEmbbeding import WordEmbedding
from word_embbeding1 import WordEmbedding


def get_fasta_seq(file):
    infile = open(file)
    spe_character = []
    sequences = []
    for line in infile:
        if (line.startswith('>') == False) and (any(x in line for x in spe_character) == False):
                sequence = ''.join(line).replace('\n', '')
                sequences.append(sequence)
    return sequences
    infile.close()


def main():
    sequences = get_fasta_seq('datasets/dataset.fasta')
    w2v = WordEmbedding(w2v=1, sg=1, ngram_len=2, vectordim= 20)
    seqs_tokenized = w2v.tokenize_sequences(sequences)
    #w2v_sequeces = w2v.sequence_preparation(filename='ubisites_caso/datasets/dataset.fasta')
    w2v.train_wordembedding(seqs_tokenized=seqs_tokenized)

if __name__ == "__main__":
    main()