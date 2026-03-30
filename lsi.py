import os
import pickle
from collections import Counter
import numpy as np
import faiss
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from index import InvertedIndexReader


class LSIRetriever:
    """
    LSI retriever backed by sparse TF-IDF, truncated SVD, and FAISS.

    The SVD step uses randomized truncated SVD from scikit-learn, which is
    suitable for large sparse term-document matrices.
    """

    def __init__(self, index_name, output_dir, postings_encoding, artifact_prefix="lsi"):
        self.index_name = index_name
        self.output_dir = output_dir
        self.postings_encoding = postings_encoding

        self.meta_path = os.path.join(output_dir, f"{artifact_prefix}_meta.pkl")
        self.model_path = os.path.join(output_dir, f"{artifact_prefix}_svd.pkl")
        self.faiss_path = os.path.join(output_dir, f"{artifact_prefix}.faiss")

        self.meta = None
        self.svd = None
        self.faiss_index = None
        self.term_to_col = None

    def _artifacts_exist(self):
        return (
            os.path.exists(self.meta_path)
            and os.path.exists(self.model_path)
            and os.path.exists(self.faiss_path)
        )

    def _set_runtime_fields(self):
        self.term_to_col = {term_id: col for col, term_id in enumerate(self.meta["term_ids"])}

    def load(self):
        with open(self.meta_path, "rb") as f:
            self.meta = pickle.load(f)

        with open(self.model_path, "rb") as f:
            self.svd = pickle.load(f)

        self.faiss_index = faiss.read_index(self.faiss_path)

        self.meta["idf"] = np.array(self.meta["idf"], dtype=np.float32)
        self._set_runtime_fields()

    def _build_sparse_tfidf(self):
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as reader:
            doc_ids = sorted(reader.doc_length.keys())
            term_ids = list(reader.terms)

            n_docs = len(doc_ids)
            n_terms = len(term_ids)
            if n_docs == 0 or n_terms == 0:
                raise ValueError("Cannot build LSI model: index has no documents or terms")

            doc_to_row = {doc_id: row for row, doc_id in enumerate(doc_ids)}

            rows = []
            cols = []
            data = []
            idf = np.zeros(n_terms, dtype=np.float32)

            for col, term_id in enumerate(term_ids):
                postings, tf_list = reader.get_postings_list(term_id)
                df = len(postings)
                idf[col] = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0

                for doc_id, tf in zip(postings, tf_list):
                    if tf <= 0:
                        continue
                    rows.append(doc_to_row[doc_id])
                    cols.append(col)
                    data.append((1.0 + np.log(float(tf))) * float(idf[col]))

        if not data:
            raise ValueError("Cannot build LSI model: TF-IDF matrix is empty")

        matrix = csr_matrix(
            (
                np.array(data, dtype=np.float32),
                (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)),
            ),
            shape=(len(doc_ids), len(term_ids)),
            dtype=np.float32,
        )

        return matrix, doc_ids, term_ids, idf

    def build(self, n_components=256, n_iter=7, random_state=42):
        matrix, doc_ids, term_ids, idf = self._build_sparse_tfidf()

        max_rank = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
        if max_rank < 1:
            raise ValueError("Cannot build LSI model: matrix rank too small")

        n_components = min(int(n_components), int(max_rank))
        if n_components < 1:
            raise ValueError("n_components must be >= 1")

        svd = TruncatedSVD(
            n_components=n_components,
            n_iter=int(n_iter),
            random_state=int(random_state),
        )

        doc_vectors = svd.fit_transform(matrix).astype(np.float32)
        norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        doc_vectors = doc_vectors / norms

        faiss_index = faiss.IndexFlatIP(doc_vectors.shape[1])
        faiss_index.add(doc_vectors)

        meta = {
            "index_name": self.index_name,
            "postings_encoding": self.postings_encoding.__name__,
            "n_components": n_components,
            "n_iter": int(n_iter),
            "doc_ids": doc_ids,
            "term_ids": term_ids,
            "idf": idf,
        }

        with open(self.meta_path, "wb") as f:
            pickle.dump(meta, f)
        with open(self.model_path, "wb") as f:
            pickle.dump(svd, f)
        faiss.write_index(faiss_index, self.faiss_path)

        self.meta = meta
        self.svd = svd
        self.faiss_index = faiss_index
        self._set_runtime_fields()

    def load_or_build(self, term_id_map, n_components=256, n_iter=7, rebuild=False):
        if not rebuild and self._artifacts_exist():
            self.load()
            if (
                self.meta.get("index_name") == self.index_name
                and self.meta.get("postings_encoding") == self.postings_encoding.__name__
                and len(self.meta.get("term_ids", [])) == len(term_id_map)
                and int(self.meta.get("n_components", -1)) == int(n_components)
            ):
                return

        self.build(n_components=n_components, n_iter=n_iter)

    def _query_vector(self, query, term_id_map):
        token_counts = Counter(query.split())
        cols = []
        vals = []

        for token, count in token_counts.items():
            term_id = term_id_map.get_id_if_exists(token)
            if term_id is None:
                continue
            col = self.term_to_col.get(term_id)
            if col is None:
                continue

            weight = (1.0 + np.log(float(count))) * float(self.meta["idf"][col])
            cols.append(col)
            vals.append(weight)

        if not cols:
            return None

        q_vec = csr_matrix(
            (
                np.array(vals, dtype=np.float32),
                (np.zeros(len(cols), dtype=np.int32), np.array(cols, dtype=np.int32)),
            ),
            shape=(1, len(self.meta["term_ids"])),
            dtype=np.float32,
        )

        return q_vec

    def search(self, query, term_id_map, doc_id_map, k=10):
        if self.faiss_index is None or self.svd is None or self.meta is None:
            self.load()

        q_vec = self._query_vector(query, term_id_map)
        if q_vec is None:
            return []

        query_latent = self.svd.transform(q_vec).astype(np.float32)
        q_norm = np.linalg.norm(query_latent, axis=1, keepdims=True)
        q_norm[q_norm == 0.0] = 1.0
        query_latent = query_latent / q_norm

        scores, rows = self.faiss_index.search(query_latent, int(k))
        results = []
        for score, row in zip(scores[0], rows[0]):
            if row < 0:
                continue
            doc_id = self.meta["doc_ids"][row]
            results.append((float(score), doc_id_map[doc_id]))

        return results
