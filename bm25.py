import math
from itertools import chain
import numpy as np


class BM25:
    """
    Best Match 25.    
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, words, pls, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.DL)
        self.AVGDL = sum(index.DL.values()) / self.N
        self.words, self.pls = words, pls

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.
        
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        
        Returns:
        -----------
        idf: dictionary of idf scores. As follows: 
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def get_top_n(self, sim_dict, N=3, score=True):
        """
            Sort and return the highest N documents according to the cosine similarity score.
            Generate a dictionary of cosine similarity scores 
        
            Parameters:
            -----------
            sim_dict: a dictionary of similarity score as follows:
                                                                        key: document id (e.g., doc_id)
                                                                        value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

            N: Integer (how many documents to retrieve). By default N = 3
            
            Returns:
            -----------
            if score:
                a ranked list of pairs (doc_id, score) in the length of N.

            else:
                a ranked list of doc_ids in the length of N.
            """
        if score:
            return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                          reverse=True)[:N]

        return sorted([(doc_id) for doc_id, _ in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]

    def _score(self, query, doc_id):
        """
        This function calculate the bm25 score for given query and document.
        
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.
        
        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        doc_len = self.index.DL[str(doc_id)]

        for term in query:
            if term in self.index.term_total.keys():
                term_frequencies = dict(self.pls[self.words.index(term)])
                if doc_id in term_frequencies.keys():
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score

    def search(self, query, N=3, return_scores=False):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query. 
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.
        
        Parameters:
        -----------
        query: list_of_tokens
        
        Returns:
        -----------
        result: list_of_top_N, sorted by relevance
        """

        search_result = dict()
        tokens = np.unique(query)
        self.idf = self.calc_idf(tokens)

        candidates = []
        scores = dict()
        for term in tokens:
            if term in self.words:
                doc_list = (self.pls[self.words.index(term)])
                candidates += [x[0] for x in doc_list]

        for doc in np.unique(candidates):
            scores[doc] = self._score(query, doc)

        search_result[id] = self.get_top_n(scores, N, return_scores)

        return search_result
