import os
import pickle
import contextlib
import heapq
import time
import math
import argparse

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, FSTTermMap, sorted_merge_posts_tfs_positions
from compression import StandardPostings, VBEPostings, EliasGammaPostings
from tqdm import tqdm
from lsi import LSIRetriever


COMPRESSION_ENCODINGS = {
    'standard': StandardPostings,
    'vbe': VBEPostings,
    'elias-gamma': EliasGammaPostings,
}


def get_postings_encoding(compression_name):
    key = compression_name.lower()
    if key not in COMPRESSION_ENCODINGS:
        raise ValueError("compression must be one of: standard, vbe, elias-gamma")
    return COMPRESSION_ENCODINGS[key]

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(FSTTermMap): Maps terms to termIDs using FST
    doc_id_map(IdMap): Maps relative document paths (e.g.,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path to collection data
    output_dir(str): Path to output index files
    postings_encoding: See compression.py (e.g., StandardPostings,
                    VBEPostings, etc.)
    index_name(str): Inverted index file name
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = FSTTermMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self._lsi_retriever = None
        self._lsi_config = None

        # Stores intermediate inverted index file names.
        self.intermediate_indices = []

    def save(self):
        """Save doc_id_map and term_id_map to output directory using pickle."""

        self.term_id_map.minimize()
        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Load doc_id_map and term_id_map from output directory."""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
            if isinstance(self.term_id_map, IdMap):
                self.term_id_map = FSTTermMap.from_id_list(self.term_id_map.id_to_str)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Parse text files into a sequence of <termID, docID, position> triples.

        Parameters
        ----------
        block_dir_relative : str
            Relative path to the directory that contains text files for one block.

            Note: one folder under collection is treated as one block.

        Returns
        -------
        List[Tuple[Int, Int, Int]]
            All term-doc-position triples extracted from the block.

        Uses self.term_id_map and self.doc_id_map to obtain termIDs and docIDs.
        These mappings persist across parse_block(...) calls.
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                for position, token in enumerate(f.read().split(), start = 1):
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname], position))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Invert td_pairs (list of <termID, docID, position> triples) and write to index.

        Uses one dictionary per block during inversion, and stores both
        sorted docIDs and their TF values.

        ASSUMPTION: td_pairs fits in memory.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int, Int]]
            List of termID-docID-position triples
        index: InvertedIndexWriter
            Disk-based inverted index associated with a block
        """
        term_positions = {}
        for term_id, doc_id, position in td_pairs:
            if term_id not in term_positions:
                term_positions[term_id] = {}
            if doc_id not in term_positions[term_id]:
                term_positions[term_id][doc_id] = []
            term_positions[term_id][doc_id].append(position)

        for term_id in sorted(term_positions.keys()):
            sorted_doc_id = sorted(list(term_positions[term_id].keys()))
            assoc_positions = [term_positions[term_id][doc_id] for doc_id in sorted_doc_id]
            assoc_tf = [len(positions) for positions in assoc_positions]
            index.append(term_id, sorted_doc_id, assoc_tf, assoc_positions)

    def merge(self, indices, merged_index):
        """
        Merge all intermediate inverted indices into one final index.

        This is the EXTERNAL MERGE SORT step.

        Uses sorted_merge_posts_tfs_positions(..) from util.

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, each
            representing an iterable intermediate inverted index.

        merged_index: InvertedIndexWriter
            InvertedIndexWriter object that stores the merged result.
        """
        # Assumes there is at least one term.
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list, positions_list = next(merged_iter) # first item
        for t, postings_, tf_list_, positions_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf_pos = sorted_merge_posts_tfs_positions(
                    list(zip(postings, tf_list, positions_list)),
                    list(zip(postings_, tf_list_, positions_list_))
                )
                postings = [doc_id for (doc_id, _, _) in zip_p_tf_pos]
                tf_list = [tf for (_, tf, _) in zip_p_tf_pos]
                positions_list = [positions for (_, _, positions) in zip_p_tf_pos]
            else:
                merged_index.append(curr, postings, tf_list, positions_list)
                curr, postings, tf_list, positions_list = t, postings_, tf_list_, positions_list_
        merged_index.append(curr, postings, tf_list, positions_list)

    def _get_query_term_ids(self, query):
        """Return term IDs that actually exist in the index vocabulary."""
        term_ids = []
        for word in query.split():
            term_id = self.term_id_map.get_id_if_exists(word)
            if term_id is not None:
                term_ids.append(term_id)
        return term_ids

    @staticmethod
    def _phrase_step(left_positions, right_positions):
        """Return right-term positions that directly follow left-term positions."""
        i, j = 0, 0
        matched = []
        while i < len(left_positions) and j < len(right_positions):
            target = left_positions[i] + 1
            if right_positions[j] == target:
                matched.append(right_positions[j])
                i += 1
                j += 1
            elif right_positions[j] < target:
                j += 1
            else:
                i += 1
        return matched

    @staticmethod
    def _min_cover_span(position_lists):
        """Return shortest span covering at least one position from each list."""
        if not position_lists or any(len(pos) == 0 for pos in position_lists):
            return None

        heap = []
        pointers = [0] * len(position_lists)
        current_max = -1

        for i, positions in enumerate(position_lists):
            value = positions[0]
            heapq.heappush(heap, (value, i))
            if value > current_max:
                current_max = value

        best_span = None
        while True:
            current_min, list_idx = heapq.heappop(heap)
            span = current_max - current_min + 1
            if best_span is None or span < best_span:
                best_span = span

            pointers[list_idx] += 1
            if pointers[list_idx] >= len(position_lists[list_idx]):
                break

            next_value = position_lists[list_idx][pointers[list_idx]]
            heapq.heappush(heap, (next_value, list_idx))
            if next_value > current_max:
                current_max = next_value

        return best_span

    @staticmethod
    def _bm25_term_score(tf, dl, avg_dl, idf, k1, b):
        denominator = tf + k1 * (1 - b + b * (dl / avg_dl))
        if denominator <= 0:
            return 0.0
        return idf * ((tf * (k1 + 1)) / denominator)

    @staticmethod
    def _bm25_term_upper_bound(max_tf, idf, min_dl, avg_dl, k1, b):
        if max_tf <= 0:
            return 0.0
        denominator = max_tf + k1 * (1 - b + b * (min_dl / avg_dl))
        if denominator <= 0:
            return 0.0
        return idf * ((max_tf * (k1 + 1)) / denominator)

    def retrieve_tfidf(self, query, k = 10):
        """
        Perform ranked retrieval using TaaT (Term-at-a-Time) TF-IDF.

        Returns top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       if tf(t, D) > 0
            = 0                        otherwise

        w(t, Q) = IDF = log (N / df(t))

        Score = sum over query terms of w(t, Q) * w(t, D)
            (without document-length normalization)

        Notes:
            1. DF(t) is available in postings_dict
            2. TF(t, D) is available in tf_list
            3. N is len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens separated by spaces

        Result
        ------
        List[(int, str)]
            List of tuples where first element is score similarity,
            second element is document name, sorted in descending score.

        Does not raise exceptions for query terms absent from collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = self._get_query_term_ids(query)
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25_taat(self, query, k = 10, k1 = 1.2, b = 0.75):
        """Baseline BM25 retrieval (TaaT), scoring all posting candidates."""
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = self._get_query_term_ids(query)
        if k <= 0 or not terms:
            return []

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            N = merged_index.collection_stats.get('num_docs', len(merged_index.doc_length))
            avg_dl = merged_index.collection_stats.get('avg_doc_length', 0.0)

            if N == 0:
                return []

            if avg_dl <= 0:
                avg_dl = 1.0

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    term_meta = merged_index.postings_dict[term]
                    df = term_meta[1]
                    idf = math.log(1 + ((N - df + 0.5) / (df + 0.5)))
                    postings, tf_list = merged_index.get_postings_list(term)

                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if tf <= 0:
                            continue

                        dl = merged_index.doc_length.get(doc_id, 0)
                        bm25_term_score = self._bm25_term_score(tf, dl, avg_dl, idf, k1, b)

                        if doc_id not in scores:
                            scores[doc_id] = 0.0
                        scores[doc_id] += bm25_term_score

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25_wand(self, query, k = 10, k1 = 1.2, b = 0.75):
        """
        BM25 retrieval with WAND Top-K.

        WAND uses per-term upper bounds to prune candidates,
        so not all documents require full BM25 scoring.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = self._get_query_term_ids(query)
        if k <= 0 or not terms:
            return []

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = merged_index.collection_stats.get('num_docs', len(merged_index.doc_length))
            avg_dl = merged_index.collection_stats.get('avg_doc_length', 0.0)
            min_dl = merged_index.collection_stats.get('min_doc_length', 0)

            if N == 0:
                return []

            if avg_dl <= 0:
                avg_dl = 1.0

            states = []
            for term in terms:
                if term not in merged_index.postings_dict:
                    continue

                term_meta = merged_index.postings_dict[term]
                df = term_meta[1]
                idf = math.log(1 + ((N - df + 0.5) / (df + 0.5)))
                postings, tf_list = merged_index.get_postings_list(term)
                if not postings:
                    continue

                # postings_dict tuple layout stores max_tf at index 5.
                max_tf = term_meta[5] if len(term_meta) >= 6 else (max(tf_list) if tf_list else 0)
                ub = self._bm25_term_upper_bound(max_tf, idf, min_dl, avg_dl, k1, b)
                states.append({
                    'postings': postings,
                    'tf_list': tf_list,
                    'idx': 0,
                    'idf': idf,
                    'ub': ub,
                })

            if not states:
                return []

            topk_heap = []  # min-heap of tuples (score, doc_id)
            threshold = 0.0

            while True:
                active_states = [s for s in states if s['idx'] < len(s['postings'])]
                if not active_states:
                    break

                active_states.sort(key = lambda s: s['postings'][s['idx']])

                ub_sum = 0.0
                pivot_pos = None
                pivot_doc = None
                for i, state in enumerate(active_states):
                    ub_sum += state['ub']
                    if ub_sum > threshold:
                        pivot_pos = i
                        pivot_doc = state['postings'][state['idx']]
                        break

                if pivot_pos is None:
                    break

                smallest_doc = active_states[0]['postings'][active_states[0]['idx']]
                if smallest_doc == pivot_doc:
                    doc_id = pivot_doc
                    dl = merged_index.doc_length.get(doc_id, 0)
                    score = 0.0

                    for state in active_states:
                        idx = state['idx']
                        postings = state['postings']
                        if idx < len(postings) and postings[idx] == doc_id:
                            tf = state['tf_list'][idx]
                            score += self._bm25_term_score(tf, dl, avg_dl, state['idf'], k1, b)
                            state['idx'] = idx + 1

                    if len(topk_heap) < k:
                        heapq.heappush(topk_heap, (score, doc_id))
                        if len(topk_heap) == k:
                            threshold = topk_heap[0][0]
                    elif score > topk_heap[0][0]:
                        heapq.heapreplace(topk_heap, (score, doc_id))
                        threshold = topk_heap[0][0]
                else:
                    for i in range(pivot_pos):
                        state = active_states[i]
                        postings = state['postings']
                        idx = state['idx']
                        while idx < len(postings) and postings[idx] < pivot_doc:
                            idx += 1
                        state['idx'] = idx

            docs = sorted(topk_heap, key = lambda x: x[0], reverse = True)
            return [(score, self.doc_id_map[doc_id]) for (score, doc_id) in docs]

    def retrieve_bm25(self, query, k = 10, k1 = 1.2, b = 0.75, use_wand = True):
        """
        Perform ranked retrieval using BM25.

        score(D, Q) = sigma_{t di Q} idf(t) * ((tf(t, D) * (k1 + 1)) /
                       (tf(t, D) + k1 * (1 - b + b * |D| / avgdl)))

        idf(t) = log(1 + ((N - df(t) + 0.5) / (df(t) + 0.5)))

        Parameters
        ----------
        query: str
            Query tokens separated by spaces
        k: int
            Number of top documents to return
        k1: float
            TF saturation parameter
        b: float
            Document-length normalization parameter
        use_wand: bool
            If True use WAND Top-K, otherwise use TaaT

        Result
        ------
        List[(float, str)]
            List of tuples where first element is score similarity,
            second element is document name.
        """
        if use_wand:
            return self.retrieve_bm25_wand(query, k = k, k1 = k1, b = b)
        return self.retrieve_bm25_taat(query, k = k, k1 = k1, b = b)

    def retrieve_phrase(self, query, k = 10):
        """
        Perform exact phrase retrieval using positional postings.

        Returns top-K docs ranked by phrase frequency.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = self._get_query_term_ids(query)
        if k <= 0 or not terms:
            return []

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            term_docs_positions = []
            for term in terms:
                if term not in merged_index.postings_dict:
                    return []
                postings, tf_list, positions_list = merged_index.get_postings_with_positions(term)
                doc_to_positions = {postings[i]: positions_list[i] for i in range(len(postings))}
                term_docs_positions.append(doc_to_positions)

            candidate = term_docs_positions[0]
            for i in range(1, len(term_docs_positions)):
                next_docs = term_docs_positions[i]
                new_candidate = {}
                for doc_id, left_positions in candidate.items():
                    if doc_id not in next_docs:
                        continue
                    matched = self._phrase_step(left_positions, next_docs[doc_id])
                    if matched:
                        new_candidate[doc_id] = matched
                candidate = new_candidate
                if not candidate:
                    return []

            docs = [(len(pos_list), self.doc_id_map[doc_id]) for doc_id, pos_list in candidate.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_proximity(self, query, k = 10, window = 5):
        """
        Perform proximity retrieval using positional postings.

        A document matches if all query terms appear within a minimum covering
        span <= window. Results are ranked by 1 / min_span.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        if k <= 0:
            return []
        if window <= 0:
            raise ValueError("window must be a positive integer")

        raw_terms = query.split()
        if len(raw_terms) < 2:
            return []

        term_ids = []
        for word in raw_terms:
            term_id = self.term_id_map.get_id_if_exists(word)
            if term_id is None:
                return []
            term_ids.append(term_id)

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            per_term_positions = []
            for term_id in term_ids:
                if term_id not in merged_index.postings_dict:
                    return []
                postings, tf_list, positions_list = merged_index.get_postings_with_positions(term_id)
                per_term_positions.append({postings[i]: positions_list[i] for i in range(len(postings))})

            candidate_docs = set(per_term_positions[0].keys())
            for doc_map in per_term_positions[1:]:
                candidate_docs &= set(doc_map.keys())
            if not candidate_docs:
                return []

            scored_docs = []
            for doc_id in candidate_docs:
                doc_position_lists = [doc_map[doc_id] for doc_map in per_term_positions]
                min_span = self._min_cover_span(doc_position_lists)
                if min_span is None or min_span > window:
                    continue
                scored_docs.append((1.0 / float(min_span), self.doc_id_map[doc_id]))

            return sorted(scored_docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_lsi(self, query, k = 10, n_components = 256, n_iter = 7, rebuild_lsi = False):
        """
        Perform retrieval using LSI document vectors indexed with FAISS.

        Notes
        -----
        - Uses sparse TF-IDF term-document matrix.
        - Uses randomized TruncatedSVD for efficient dimensionality reduction.
        - Stores FAISS index artifacts in output_dir for reuse.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        config = (int(n_components), int(n_iter))
        needs_new_retriever = (
            self._lsi_retriever is None
            or self._lsi_config != config
        )
        if needs_new_retriever:
            self._lsi_retriever = LSIRetriever(
                index_name = self.index_name,
                output_dir = self.output_dir,
                postings_encoding = self.postings_encoding,
            )
            self._lsi_config = config

        self._lsi_retriever.load_or_build(
            term_id_map = self.term_id_map,
            n_components = n_components,
            n_iter = n_iter,
            rebuild = rebuild_lsi,
        )

        return self._lsi_retriever.search(
            query = query,
            term_id_map = self.term_id_map,
            doc_id_map = self.doc_id_map,
            k = k,
        )

    def retrieve(self, query, k = 10, scoring = 'tfidf', **kwargs):
        """
        Retrieval wrapper for selecting scoring scheme.

        Parameters
        ----------
        query: str
            Query tokens separated by spaces
        k: int
            Number of top documents to return
        scoring: str
            Scoring scheme: 'tfidf', 'bm25', 'phrase', 'proximity', or 'lsi'
        kwargs:
            Extra parameters for specific schemes (e.g., k1, b for BM25,
            window for proximity search, n_components and n_iter for LSI)
        """
        scoring = scoring.lower()
        if scoring == 'tfidf':
            return self.retrieve_tfidf(query, k = k)
        if scoring == 'bm25':
            return self.retrieve_bm25(query,
                                      k = k,
                                      k1 = kwargs.get('k1', 1.2),
                                      b = kwargs.get('b', 0.75),
                                      use_wand = kwargs.get('use_wand', True))
        if scoring == 'phrase':
            return self.retrieve_phrase(query, k = k)
        if scoring == 'proximity':
            return self.retrieve_proximity(query,
                                           k = k,
                                           window = kwargs.get('window', 5))
        if scoring == 'lsi':
            return self.retrieve_lsi(query,
                                     k = k,
                                     n_components = kwargs.get('lsi_components', 256),
                                     n_iter = kwargs.get('lsi_n_iter', 7),
                                     rebuild_lsi = kwargs.get('rebuild_lsi', False))
        raise ValueError("scoring must be 'tfidf', 'bm25', 'phrase', 'proximity', or 'lsi'")

    def index(self):
        """
        Base indexing code
        Main routine for BSBI (blocked-sort based indexing).

        Scans all collection data, calls parse_block for parsing,
        and calls invert_write for block-level inversion.
        """
        self.intermediate_indices = []

        # Loop over each sub-directory in collection (one block per folder)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


class SPIMIIndex(BSBIIndex):
    """SPIMI indexer that reuses retrieval and merge logic from BSBIIndex."""

    def _write_term_dict(self, term_dict, index):
        """Write in-memory term dictionary in sorted termID order."""
        for term_id in sorted(term_dict.keys()):
            doc_positions = term_dict[term_id]
            sorted_doc_id = sorted(doc_positions.keys())
            assoc_positions = [doc_positions[doc_id] for doc_id in sorted_doc_id]
            assoc_tf = [len(positions) for positions in assoc_positions]
            index.append(term_id, sorted_doc_id, assoc_tf, assoc_positions)

    def _flush_spimi_run(self, term_dict, run_id):
        """Flush one SPIMI run to disk and clear in-memory dictionary."""
        if not term_dict:
            return run_id

        index_id = f"intermediate_index_spimi_{run_id}"
        self.intermediate_indices.append(index_id)

        with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
            self._write_term_dict(term_dict, index)

        term_dict.clear()
        return run_id + 1

    def _iter_block_documents(self, block_dir_relative):
        """Yield canonical relative document paths for one block."""
        block_dir = "./" + self.data_dir + "/" + block_dir_relative
        for filename in sorted(next(os.walk(block_dir))[2]):
            yield block_dir + "/" + filename

    def index(self, max_terms_in_memory = 50000):
        """
        Main routine for SPIMI (single-pass in-memory indexing).

        Parameters
        ----------
        max_terms_in_memory: int
            Maximum number of unique terms kept in memory before a flush.
            Use a larger value if more RAM is available.
        """
        if max_terms_in_memory is not None and max_terms_in_memory <= 0:
            raise ValueError("max_terms_in_memory must be a positive integer")

        self.intermediate_indices = []
        term_dict = {}
        run_id = 1

        block_dirs = sorted(next(os.walk(self.data_dir))[1])
        for block_dir_relative in tqdm(block_dirs):
            for docname in self._iter_block_documents(block_dir_relative):
                doc_id = self.doc_id_map[docname]
                with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                    for position, token in enumerate(f.read().split(), start = 1):
                        term_id = self.term_id_map[token]

                        should_flush = (
                            max_terms_in_memory is not None
                            and term_id not in term_dict
                            and len(term_dict) >= max_terms_in_memory
                        )
                        if should_flush:
                            run_id = self._flush_spimi_run(term_dict, run_id)

                        if term_id not in term_dict:
                            term_dict[term_id] = {}

                        if doc_id not in term_dict[term_id]:
                            term_dict[term_id][doc_id] = []
                        term_dict[term_id][doc_id].append(position)

        self._flush_spimi_run(term_dict, run_id)
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            if not self.intermediate_indices:
                return

            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory = self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build inverted index (SPIMI or BSBI)')
    parser.add_argument('--compression', default='elias-gamma',
                        choices=['standard', 'vbe', 'elias-gamma'],
                        help='Postings compression type')
    parser.add_argument('--indexing-mode', default='spimi',
                        choices=['spimi', 'bsbi'],
                        help='Index construction algorithm')
    parser.add_argument('--spimi-max-terms', type=int, default=50000,
                        help='Maximum number of unique terms kept in-memory per SPIMI run')
    args = parser.parse_args()

    indexer_class = SPIMIIndex if args.indexing_mode == 'spimi' else BSBIIndex
    indexer = indexer_class(data_dir = 'collection', \
                            postings_encoding = get_postings_encoding(args.compression), \
                            output_dir = 'index')

    if args.indexing_mode == 'spimi':
        indexer.index(max_terms_in_memory = args.spimi_max_terms)
    else:
        indexer.index()
