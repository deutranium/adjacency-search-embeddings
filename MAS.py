import json
import networkx as nx
import random
from gensim.models import Word2Vec
import pickle


class MAS:
    def __init__(
        self,
        g,
        k,
        num_permutations,
        thresh_len,
        # threshold,
        dimensions,
        window_size,
        workers,
        iter,
        min_thresh=1,
        seed=1,
    ) -> None:
        self.g = g
        self.k = k
        self.num_permutations = num_permutations
        self.thresh_len = thresh_len
        self.min_thresh = min_thresh

        # for Word2Vec
        self.dimensions = dimensions
        self.window_size = window_size
        self.workers = workers
        self.iter = iter

        self.SEED = seed

        self.walks = self.get_sentences()
        self.model = self.get_model()

    def get_sentences(self):
        g = self.g
        k = self.k
        permutations = self.num_permutations

        nodes = list(g.nodes())
        sentences = []

        for p in range(permutations):
            print(f"PERMUTATION: {p}")

            for node in nodes:
                neighbours = list(nx.all_neighbors(g, node))
                random.shuffle(neighbours)
                len_n = len(neighbours)

                if len_n:
                    for i in range(0, len_n, k):
                        start = min(i, max(0, len_n - k))
                        this_sentence = [node] + neighbours[start : start + k]
                        this_sentence = self.get_MAS(this_sentence)

                        sentences.append(this_sentence)
                else:
                    sentences.append([node])

        return sentences
    
    def get_MAS(self, sentence):
        """
        Get the MAS sequence from a given set of activated node and its k neighbours
        """
        thresh_len = self.thresh_len
        idx = 0
        count_dict = {} # store current adjacencies
        cur_nodes = set(sentence)
        g = self.g

        res = sentence

        # initialise count_dict
        for node in sentence:
            this_neighbours = list(nx.all_neighbors(g, node))
            for n in this_neighbours:
                if n not in cur_nodes:
                    if n not in count_dict:
                        count_dict[n] = 0
                    count_dict[n] += 1

        if not count_dict:
            return sentence

        while idx < thresh_len:
            this_max = max(count_dict, key=count_dict.get)
            res.append(this_max)
            cur_nodes.add(this_max)
            idx += 1

            this_neighbours = list(nx.all_neighbors(g, this_max))
            for neigh in this_neighbours:
                if neigh not in cur_nodes:
                    if neigh not in count_dict:
                        count_dict[neigh] = 0
                    count_dict[neigh] += 1

        return res
            

    def get_model(self):
        """
        Get Work2Vec model
        """
        model = Word2Vec(
            self.walks,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=self.workers,
            epochs=self.iter,
            seed=self.SEED,
        )
        return model

    def get_embedding(self, u):
        idx = self.model.wv.key_to_index[u]
        return self.model.wv[idx]
