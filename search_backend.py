import numpy as np

from inverted_index_colab import InvertedIndex
from collections import defaultdict, Counter
from nltk.corpus import stopwords
import re
from tfidf import get_topn_for_query
from bm25 import BM25

"""
Indexes AND PATHS
"""

BUCKET_NAME = '../small_indices'
PAGE_RANK_PATH = 'part-00000-2cc0993d-c70b-4b4c-9d86-83eed6e2fb0e-c000.csv.gz'
PAGE_VIEW_PATH = None
DOC_TITLES_PATH = 'doc_titles'
BODY_INDEX_PATH = 'text_index/index'
TITLE_INDEX_PATH = 'title_index/index'
ANCHOR_INDEX_PATH = 'anchor_index/index'

CORPUS_STOP_WORDS = ["category", "references", "also", "external", "links",
                     "may", "first", "see", "history", "people", "one", "two",
                     "part", "thumb", "including", "second", "following",
                     "many", "however", "would", "became"]
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


class SearchMaster:
    def __init__(self) -> None:
        self.page_rank = InvertedIndex.read_csv(BUCKET_NAME, PAGE_RANK_PATH)  # dict {id:float}
        self.page_views = None  # InvertedIndex.read_index(BUCKET_NAME, PAGE_VIEW_PATH)  # dict {id:int}
        self.title_index = InvertedIndex.read_index(BUCKET_NAME, TITLE_INDEX_PATH)  # II
        self.body_index = InvertedIndex.read_index(BUCKET_NAME, BODY_INDEX_PATH)  # II
        self.anchor_index = InvertedIndex.read_index(BUCKET_NAME, ANCHOR_INDEX_PATH)  # II
        self.stop_words = self.get_stop_words(CORPUS_STOP_WORDS)  # list
        self.titles = InvertedIndex.read_index(BUCKET_NAME, DOC_TITLES_PATH)  # dict {id:title}

    @staticmethod
    def get_stop_words(corpus_stopwords):
        english_stopwords = frozenset(stopwords.words('english'))
        all_stopwords = english_stopwords.union(corpus_stopwords)

        return all_stopwords

    def get_pagerank(self, wiki_ids):
        page_rank = self.page_rank
        result = []
        for doc_id in wiki_ids:
            if doc_id in page_rank.keys():
                rank = self.page_rank[str(doc_id)]
            else:
                rank = 0
            result.append(rank)

        return result

    def get_pageviews(self, wiki_ids):
        page_views = self.page_views
        result = []
        for doc_id in wiki_ids:
            if doc_id in page_views.keys():
                views = self.page_views[str(doc_id)]
            else:
                views = 0
            result.append(views)

        return result

    @staticmethod
    def get_posting_iter(index):
        """
        This function returning the iterator working with posting list.
        
        Parameters:
        ----------
        index: inverted index    
        """
        words, pls = zip(*index.posting_lists_iter())
        return words, pls

    def tokenize(self, text):
        """
        This function aims in tokenize a text into a list of tokens. Moreover, filter stopwords.
        
        Parameters:
        -----------
        text: string , representing the text to tokenize.
        
        Returns:
        -----------
        list of tokens (e.g., list of tokens).
        """
        list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                          token.group() not in self.stop_words]
        return list_of_tokens

    def result_w_title(self, doc_ids):
        # add titles
        titles = self.titles
        result = []
        for doc_id in doc_ids:
            result.append((doc_id, titles[doc_id]))

        return result

    def get_relevant_docs(self, query, index) -> Counter:
        tokenized_query = self.tokenize(query)

        words, pls = self.get_posting_iter(index)

        term_counter = Counter()
        for term in np.unique(tokenized_query):
            if term in words:
                for doc_id, tf in pls[term]:
                    term_counter[doc_id] += 1

        return term_counter

    def get_relevant_titles(self, query) -> list:
        term_counter = self.get_relevant_docs(query, self.title_index)
        return self.result_w_title(sorted(term_counter, key=term_counter.get, reverse=True))

    def get_relevant_anchors(self, query) -> list:
        term_counter = self.get_relevant_docs(query, self.anchor_index)
        return self.result_w_title(sorted(term_counter, key=term_counter.get, reverse=True))

    def body_search(self, query, n_results=100, return_scores=False):
        # tokenize query
        tokenized_query = self.tokenize(query)
        # get top 100 results
        body_index = self.body_index
        words, pls = self.get_posting_iter(body_index)
        top_n = get_topn_for_query(query=tokenized_query, index=body_index, words=words, pls=pls, N=n_results, return_scores=return_scores)

        return top_n

    def get_top_text(self, query):
        top_n = self.body_search(query)
        return self.result_w_title(top_n)

    def merge_results(self, query, title_weight=0.5, text_weight=0.5):
        """
        This function merge and sort documents retrieved by its weighte score (e.g., title and body).

        Parameters:
        -----------
        title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                                key: query_id
                                                                                value: list of pairs in the following format:(doc_id,score)

        body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                                key: query_id
                                                                                value: list of pairs in the following format:(doc_id,score)
        title_weight: float, for weigted average utilizing title and body scores
        text_weight: float, for weigted average utilizing title and body scores
        N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

        Returns:
        -----------    title_scores.update((key, value * 2) for key, value in my_dict.items())

        dictionary of querires and topN pairs as follows:
                                                            key: query_id
                                                            value: list of pairs in the following format:(doc_id,score).
        """
        # YOUR CODE HERE

        title_scores = self.get_relevant_docs(query, self.title_index)
        body_scores = self.body_search(query, self.title_index, return_scores=True)

        all_n = {}
        for key, value in body_scores:  # (wiki_id , score)
            all_n[key] = text_weight * value + title_weight * title_scores[key]

        all_n = sorted(all_n.items(), key=lambda x: x[1], reverse=True)
        return all_n

    def basic_search(self, query):
        top_n = self.merge_results(query)
        return self.result_w_title(top_n)
