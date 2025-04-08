import numpy as np
import pandas as pd
import os 
from collections import defaultdict


def co_occurrence_matrix(corpus, window_size=1):
    """
    Calculates the co-occurrence matrix for a given corpus.

    Args:
        corpus (list of str): A list of sentences or documents.
        window_size (int): The number of words to consider to the left and right of the target word.

    Returns:
        pandas.DataFrame: The co-occurrence matrix.
    """
    word_counts = defaultdict(int)
    co_occurrence_counts = defaultdict(lambda: defaultdict(int))

    for sentence in corpus:
        words = sentence.lower().split()
        for i, word in enumerate(words):
            word_counts[word] += 1
            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if i != j:
                    neighbor_word = words[j]
                    co_occurrence_counts[word][neighbor_word] += 1

    all_words = sorted(word_counts.keys())
    df = pd.DataFrame(index=all_words, columns=all_words).fillna(0)

    for word, neighbors in co_occurrence_counts.items():
        for neighbor, count in neighbors.items():
            df.loc[word, neighbor] = count

    return df

if __name__ == '__main__':
    corpus = [open(os.path.join("data",f)).read() for f in os.listdir("data")]

    co_matrix = co_occurrence_matrix(corpus, window_size=7)
    