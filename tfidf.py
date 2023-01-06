import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import math


def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters: ----------- query_to_search: list of tokens (str). This list will be preprocessed in advance (
    e.g., lower case, filtering stopwords, etc.'). Example: 'Hello, I love information retrival' --->  ['hello',
    'love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    total_vocab_size = len(index.term_total)
    query_vector = np.zeros(total_vocab_size)
    term_vector = list(index.term_total.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(index.DL)) / (df + epsilon), 10)  # smoothing

            try:
                ind = term_vector.index(token)
                query_vector[ind] = tf * idf
            except():
                pass
    return query_vector


def get_posting_iter(index):
    """
    This function returning the iterator working with posting list.

    Parameters:
    ----------
    index: inverted index
    """
    words, pls = zip(*index.posting_lists_iter())
    return words, pls


def get_candidate_documents_and_scores(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go
    through every token in query_to_search and fetch the corresponding information (e.g., term frequency,
    document frequency, etc.') needed to calculate TF-IDF from the posting list. Then it will populate the
    dictionary 'candidates.' For calculation of IDF, use log with base 10. tf will be normalized based on the
    length of the document.

    Parameters: ----------- query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g.,
    lower case, filtering stopwords, etc.'). Example: 'Hello, I love information retrival' --->  ['hello','love',
    'information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                            key: pair (doc_id,term)
                                                            value: tfidf score.
    """
    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            normlized_tfidf = [
                (doc_id, (freq / index.DL[str(doc_id)]) * math.log(len(index.DL) / index.df[term], 10)) for
                doc_id, freq in list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    """
    Generate a DataFrame `df` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.

    Parameters: ----------- query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g.,
    lower case, filtering stopwords, etc.'). Example: 'Hello, I love information retrival' --->  ['hello','love',
    'information','retrieval']

    index:           inverted index loaded from the corresponding files.


    words,pls: iterator for working with posting.

    Returns:
    -----------
    DataFrame of tfidf scores.
    """

    total_vocab_size = len(index.term_total)
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words,
                                                           pls)  # We do not need to utilize all document. Only the
    # docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    df = np.zeros((len(unique_candidates), total_vocab_size))
    df = pd.DataFrame(df)

    df.index = unique_candidates
    df.columns = index.term_total.keys()

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        df.loc[doc_id][term] = tfidf

    return df


def cosine_similarity(D, Q):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """

    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    D['sim_score'] = D.apply(lambda row: cos_sim(row, Q), axis=1)
    return D['sim_score'].to_dict()


def get_top_n(sim_dict, N=3, score=True):
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
        return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                      reverse=True)[:N]

    return sorted([(doc_id) for doc_id, _ in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


def get_topn_for_query(query, index, words, pls, N=3, return_scores=False):
    """
    Generate top n results for query

    Parameters:
    -----------
    query: list of tokens.
    index: inverted index loaded from the corresponding files.
    N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    return: list of N doc_ids, sorted by similarity in ascending order.
    """

    Q = generate_query_tfidf_vector(query, index)
    D = generate_document_tfidf_matrix(query, index, words, pls)

    sim_dict = cosine_similarity(D, Q)
    top_n = get_top_n(sim_dict, N, return_scores)

    return top_n
