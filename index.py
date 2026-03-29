import pickle
import os

class InvertedIndex:
    """
    Implements efficient scanning/reading of an Inverted Index stored in a file,
    and provides a mechanism to write the index to storage during indexing.

    Attributes
    ----------
    postings_dict: Dictionary mapping:

            termID -> (start_position_in_index_file,
                       number_of_postings_in_list,
                       length_in_bytes_of_postings_list,
                       length_in_bytes_of_tf_list,
                       max_tf_in_postings_list)

        postings_dict is the in-memory dictionary component of the
        Inverted Index.

          It maps term IDs (integers) to a 5-tuple:
              1. start_position_in_index_file: byte offset where postings are stored.
              2. number_of_postings_in_list: document frequency for the term.
              3. length_in_bytes_of_postings_list: postings list length in bytes.
              4. length_in_bytes_of_tf_list: TF list length in bytes.
              5. max_tf_in_postings_list: maximum TF value in that term postings list.

    terms: List[int]
        Ordered list of term IDs inserted into the Inverted Index.

    """
    def __init__(self, index_name, postings_encoding, directory=''):
        """
        Parameters
        ----------
        index_name (str): Name used for index files.
        postings_encoding: See compression.py (e.g., StandardPostings,
                        GapBasedPostings, etc.).
        directory (str): Directory where index files are stored.
        """

        self.index_file_path = os.path.join(directory, index_name+'.index')
        self.metadata_file_path = os.path.join(directory, index_name+'.dict')

        self.postings_encoding = postings_encoding
        self.directory = directory

        self.postings_dict = {}
        self.terms = []         # Tracks insertion order of term IDs.
        self.doc_length = {}    # key: doc ID (int), value: document length (number of tokens)
                    # Used for score normalization in TF-IDF and BM25.
        self.collection_stats = {
            'total_doc_length': 0,
            'num_docs': 0,
            'avg_doc_length': 0.0,
            'min_doc_length': 0,
        }

    def __enter__(self):
        """
        Load all metadata when entering the context.
        Metadata:
            1. Dictionary ---> postings_dict
            2. Iterator over ordered term list ---> term_iter
            3. doc_length, a dictionary with key = doc ID and
               value = number of tokens in the document.

        Metadata is stored using the "pickle" library.

        See also:

        https://docs.python.org/3/reference/datamodel.html#object.__enter__
        """
        # Open index file.
        self.index_file = open(self.index_file_path, 'rb+')

        # Load postings dictionary and term iterator from metadata file.
        with open(self.metadata_file_path, 'rb') as f:
            metadata = pickle.load(f)
            
            if len(metadata) == 3:
                self.postings_dict, self.terms, self.doc_length = metadata
                total_doc_length = sum(self.doc_length.values())
                num_docs = len(self.doc_length)
                avg_doc_length = (total_doc_length / num_docs) if num_docs > 0 else 0.0
                min_doc_length = min(self.doc_length.values()) if num_docs > 0 else 0
                self.collection_stats = {
                    'total_doc_length': total_doc_length,
                    'num_docs': num_docs,
                    'avg_doc_length': avg_doc_length,
                    'min_doc_length': min_doc_length,
                }
            else:
                self.postings_dict, self.terms, self.doc_length, self.collection_stats = metadata
                total_doc_length = sum(self.doc_length.values())
                num_docs = len(self.doc_length)
                self.collection_stats.setdefault('total_doc_length', total_doc_length)
                self.collection_stats.setdefault('num_docs', num_docs)
                self.collection_stats.setdefault('avg_doc_length', (total_doc_length / num_docs) if num_docs > 0 else 0.0)
                self.collection_stats.setdefault('min_doc_length', min(self.doc_length.values()) if num_docs > 0 else 0)

            self.term_iter = self.terms.__iter__()

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Close index file and save metadata on context exit."""
        # Close index file.
        self.index_file.close()

        total_doc_length = sum(self.doc_length.values())
        num_docs = len(self.doc_length)
        min_doc_length = min(self.doc_length.values()) if num_docs > 0 else 0
        self.collection_stats = {
            'total_doc_length': total_doc_length,
            'num_docs': num_docs,
            'avg_doc_length': (total_doc_length / num_docs) if num_docs > 0 else 0.0,
            'min_doc_length': min_doc_length,
        }

        # Save metadata to file using pickle.
        with open(self.metadata_file_path, 'wb') as f:
            pickle.dump([self.postings_dict, self.terms, self.doc_length, self.collection_stats], f)


class InvertedIndexReader(InvertedIndex):
    """
    Implements efficient scan/read operations for an Inverted Index stored in a file.
    """
    def __iter__(self):
        return self

    def reset(self):
        """
        Reset the file pointer and the term iterator to the beginning.
        """
        self.index_file.seek(0)
        self.term_iter = self.terms.__iter__() # reset term iterator

    def __next__(self):
        """
        Return the next (term, postings_list, tf_list) triple.

        This method should read only the required slice from disk and must
        not load the entire index into memory.
        """
        curr_term = next(self.term_iter)
        term_meta = self.postings_dict[curr_term]
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = term_meta[:4]
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (curr_term, postings_list, tf_list)

    def get_postings_list(self, term):
        """
        Return a postings list and its TF list for a term as
        (postings_list, tf_list).

        This method directly seeks to byte offsets and must not scan the
        whole index sequentially.
        """
        term_meta = self.postings_dict[term]
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = term_meta[:4]
        self.index_file.seek(pos)
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (postings_list, tf_list)


class InvertedIndexWriter(InvertedIndex):
    """
    Implements efficient write operations for an Inverted Index stored in a file.
    """
    def __enter__(self):
        self.index_file = open(self.index_file_path, 'wb+')
        return self

    def append(self, term, postings_list, tf_list):
        """
                Append a term, its postings_list, and TF list to the end of index file.

                This method:
                1. Encodes postings_list using self.postings_encoding.encode,
                2. Encodes tf_list using self.postings_encoding.encode_tf,
                3. Updates metadata in self.terms, self.postings_dict, and self.doc_length,
                4. Writes encoded postings and TF bytes to disk.

        Parameters
        ----------
        term:
                        A term or termID that uniquely identifies a term
        postings_list: List[Int]
                        List of docIDs where the term appears
        tf_list: List[Int]
                        List of term frequencies
        """
        self.terms.append(term) # update self.terms

        # update self.doc_length
        for i in range(len(postings_list)):
            doc_id, freq = postings_list[i], tf_list[i]
            if doc_id not in self.doc_length:
                self.doc_length[doc_id] = 0
                self.collection_stats['num_docs'] += 1
            self.doc_length[doc_id] += freq
            self.collection_stats['total_doc_length'] += freq

        if self.collection_stats['num_docs'] > 0:
            self.collection_stats['avg_doc_length'] = self.collection_stats['total_doc_length'] / self.collection_stats['num_docs']
            self.collection_stats['min_doc_length'] = min(self.doc_length.values())
        else:
            self.collection_stats['avg_doc_length'] = 0.0
            self.collection_stats['min_doc_length'] = 0

        self.index_file.seek(0, os.SEEK_END)
        curr_position_in_byte = self.index_file.tell()
        compressed_postings = self.postings_encoding.encode(postings_list)
        compressed_tf_list = self.postings_encoding.encode_tf(tf_list)
        self.index_file.write(compressed_postings)
        self.index_file.write(compressed_tf_list)
        max_tf = max(tf_list) if tf_list else 0
        self.postings_dict[term] = (curr_position_in_byte, len(postings_list), \
                        len(compressed_postings), len(compressed_tf_list), max_tf)


if __name__ == "__main__":

    from compression import VBEPostings

    with InvertedIndexWriter('test', postings_encoding=VBEPostings, directory='./tmp/') as index:
        index.append(1, [2, 3, 4, 8, 10], [2, 4, 2, 3, 30])
        index.append(2, [3, 4, 5], [34, 23, 56])
        index.index_file.seek(0)
        assert index.terms == [1,2], "terms are incorrect"
        assert index.doc_length == {2:2, 3:38, 4:25, 5:56, 8:3, 10:30}, "doc_length is incorrect"
        assert index.collection_stats == {'total_doc_length': 154, 'num_docs': 6, 'avg_doc_length': 154 / 6, 'min_doc_length': 2}, "collection_stats is incorrect"
        assert index.postings_dict == {1: (0, \
                                           5, \
                                           len(VBEPostings.encode([2,3,4,8,10])), \
                                           len(VBEPostings.encode_tf([2,4,2,3,30])), \
                                           30),
                                       2: (len(VBEPostings.encode([2,3,4,8,10])) + len(VBEPostings.encode_tf([2,4,2,3,30])), \
                                           3, \
                                           len(VBEPostings.encode([3,4,5])), \
                                           len(VBEPostings.encode_tf([34,23,56])), \
                                           56)}, "postings dictionary is incorrect"
        
        index.index_file.seek(index.postings_dict[2][0])
        assert VBEPostings.decode(index.index_file.read(len(VBEPostings.encode([3,4,5])))) == [3,4,5], "there is an error"
        assert VBEPostings.decode_tf(index.index_file.read(len(VBEPostings.encode_tf([34,23,56])))) == [34,23,56], "there is an error"
