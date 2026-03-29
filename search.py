import argparse

from bsbi import BSBIIndex, get_postings_encoding


def parse_args():
    parser = argparse.ArgumentParser(description='Search documents with a BSBI index')
    parser.add_argument('--compression', default='elias-gamma',
                        choices=['standard', 'vbe', 'elias-gamma'],
                        help='Postings compression type used when reading the index')
    parser.add_argument('--scoring', default='bm25', choices=['tfidf', 'bm25'],
                        help='Retrieval scoring scheme')
    parser.add_argument('--bm25', default='wand', choices=['wand', 'taat'],
                        help='BM25 retrieval mode')
    parser.add_argument('-k', type=int, default=10, help='Top-K documents')
    return parser.parse_args()


def main():
    args = parse_args()

    # Indexing is assumed to be done beforehand.
    # BSBIIndex acts as an abstraction over the stored index.
    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = get_postings_encoding(args.compression), \
                              output_dir = 'index')

    queries = ["alkylated with radioactive iodoacetate", \
               "psychodrama for disturbed children", \
               "lipid metabolism in toxemia and normal pregnancy"]
               
    for query in queries:
        print("Query  : ", query)
        print("Results:")
        use_wand = (args.bm25 == 'wand')
        for (score, doc) in BSBI_instance.retrieve(query,
                                                   k = args.k,
                                                   scoring = args.scoring,
                                                   use_wand = use_wand):
            print(f"{doc:30} {score:>.3f}")
        print()


if __name__ == '__main__':
    main()