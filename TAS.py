import json
import networkx as nx
import random
from gensim.models import Word2Vec
import pickle


class TAS:
    def __init__(
        self,
        g,
        k,
        num_permutations,
        thresh_len, # walk_len
        threshold,
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
        self.thresh = threshold
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
                        this_sentence = [node] + neighbours[start : start + k] # getting neighbours

                        node_thresh = min(self.thresh, g.degree(node)) # a node can't have a threshold higher than its degree

                        for t in range(self.min_thresh, node_thresh + 1):
                            this_threshold_sentence = self.get_threshold_sequence(
                                this_sentence, t
                            )
                            sentences.append(this_threshold_sentence)
                else:
                    sentences.append([node])

        return sentences

    def get_threshold_sequence(self, nodes, t):
        """
        Get a sequence from a list of selected `nodes` and a threshold `t`
        """

        threshold_count = {node: t for node in nodes}
        activated = {node: 0 for node in nodes}

        idx = 0

        while idx < len(activated) <= self.thresh_len:
            k = list(activated.keys())
            this_node = k[idx]

            this_neighbours = list(self.g.neighbors(this_node))
            random.shuffle(this_neighbours)

            for neighbour in this_neighbours:
                if neighbour in threshold_count:
                    threshold_count[neighbour] += 1
                else:
                    threshold_count[neighbour] = 1

                if threshold_count[neighbour] == t:
                    activated[neighbour] = 0

            idx += 1
        return list(activated.keys())

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

    def store_embeddings(self, path):
        d = self.model.wv.key_to_index
        embeds = self.model.wv

        with open(f"{path}_d_TAS.json", "w", encoding="utf-8") as f:
            json.dump(d, f, indent=4)
        with open(f"{path}_embeds_TAS.pickle", "wb") as handle:
            pickle.dump(embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)
