import argparse
import re
import math
from bsbi import BSBIIndex, get_postings_encoding

######## >>>>> an IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ Compute search effectiveness score using
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
        Binary vector such as [1, 0, 1, 1, 1, 0]
        representing relevance at rank 1, 2, 3, and so on.
        
      Returns
      -------
      Float
        RBP score
  """
  score = 0.
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score


def dcg(ranking):
  """ Compute search effectiveness score using
      Discounted Cumulative Gain (DCG)

      Parameters
      ----------
      ranking: List[int]
        Binary relevance vector for ranked documents

      Returns
      -------
      Float
        DCG score
  """
  score = 0.
  for i, rel in enumerate(ranking, start = 1):
    if i == 1:
      score += rel
    else:
      score += rel / math.log2(i + 1)
  return score


def ndcg(ranking, num_relevant = None):
  """ Compute search effectiveness score using
      Normalized Discounted Cumulative Gain (NDCG)

      Parameters
      ----------
      ranking: List[int]
        Binary relevance vector for ranked documents
      num_relevant: int or None
        Total number of relevant documents for the query.
        If None, it is computed from ranking.

      Returns
      -------
      Float
        NDCG score
  """
  actual_dcg = dcg(ranking)

  if num_relevant is None:
    num_relevant = sum(ranking)

  ideal_ones = min(num_relevant, len(ranking))
  ideal_ranking = [1] * ideal_ones + [0] * (len(ranking) - ideal_ones)
  ideal_dcg = dcg(ideal_ranking)

  if ideal_dcg == 0:
    return 0.
  return actual_dcg / ideal_dcg


def ap(ranking, num_relevant):
  """ Compute search effectiveness score using
      Average Precision (AP)

      Parameters
      ----------
      ranking: List[int]
        Binary relevance vector for ranked documents
      num_relevant: int
        Total number of relevant documents for the query

      Returns
      -------
      Float
        AP score
  """
  if num_relevant == 0:
    return 0.

  rel_so_far = 0
  precision_sum = 0.
  for i, rel in enumerate(ranking, start = 1):
    if rel == 1:
      rel_so_far += 1
      precision_sum += rel_so_far / i

  return precision_sum / num_relevant


######## >>>>> load qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ Load query relevance judgment (qrels)
      in dictionary-of-dictionaries format.
      qrels[query id][document id]

      Example: qrels["Q3"][12] = 1 means Doc 12
      is relevant to Q3; qrels["Q3"][10] = 0 means
      Doc 10 is not relevant to Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUATION

def eval(qrels, query_file = "queries.txt", k = 1000, scoring = 'tfidf', k1 = 1.2, b = 0.75,
         compression = 'elias-gamma', bm25_retrieval = 'wand',
         lsi_components = 256, lsi_n_iter = 7, rebuild_lsi = False):
  """ 
    Loop through all 30 queries, compute per-query metrics,
    then compute mean scores over all queries.
    For each query, return top-k documents.
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = get_postings_encoding(compression), \
                          output_dir = 'index')

  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ndcg_scores = []
    ap_scores = []
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])
      num_relevant = sum(qrels[qid].values())

      # Note: doc IDs during indexing may differ from IDs in qrels file paths.
      ranking = []
      use_wand = (bm25_retrieval == 'wand')
      for (score, doc) in BSBI_instance.retrieve(query,
                                                 k = k,
                                                 scoring = scoring,
                                                 k1 = k1,
                                                 b = b,
                                                 use_wand = use_wand,
                                                 lsi_components = lsi_components,
                                                 lsi_n_iter = lsi_n_iter,
                                                 rebuild_lsi = rebuild_lsi):
        did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
        ranking.append(qrels[qid][did])
      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking))
      ndcg_scores.append(ndcg(ranking, num_relevant = num_relevant))
      ap_scores.append(ap(ranking, num_relevant = num_relevant))

  print(f"{scoring.upper()} evaluation results on 30 queries (bm25_retrieval={bm25_retrieval})")
  print("RBP score =", sum(rbp_scores) / len(rbp_scores))
  print("DCG score =", sum(dcg_scores) / len(dcg_scores))
  print("NDCG score =", sum(ndcg_scores) / len(ndcg_scores))
  print("AP score =", sum(ap_scores) / len(ap_scores))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluate retrieval metrics')
  parser.add_argument('--compression', default='elias-gamma',
                      choices=['standard', 'vbe', 'elias-gamma'],
                      help='Postings compression type used when reading the index')
  parser.add_argument('--scoring', default='all', choices=['all', 'tfidf', 'bm25', 'lsi'],
                      help='Scoring scheme to evaluate')
  parser.add_argument('-k', type=int, default=1000, help='Top-K documents per query for evaluation')
  parser.add_argument('--k1', type=float, default=1.2, help='BM25 k1 parameter')
  parser.add_argument('--b', type=float, default=0.75, help='BM25 b parameter')
  parser.add_argument('--bm25', default='wand', choices=['wand', 'taat'],
                      help='BM25 retrieval mode')
  parser.add_argument('--lsi-components', type=int, default=256,
                      help='Number of latent dimensions for LSI evaluation')
  parser.add_argument('--lsi-n-iter', type=int, default=7,
                      help='Randomized SVD power iterations for LSI evaluation')
  parser.add_argument('--rebuild-lsi', action='store_true',
                      help='Force rebuild of LSI model before evaluation')
  args = parser.parse_args()

  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels is incorrect"
  assert qrels["Q1"][300] == 0, "qrels is incorrect"

  if args.scoring in ['all', 'tfidf']:
    eval(qrels,
         k = args.k,
         scoring = 'tfidf',
         compression = args.compression,
         bm25_retrieval = args.bm25)
  if args.scoring in ['all', 'bm25']:
    eval(qrels,
         k = args.k,
         scoring = 'bm25',
         k1 = args.k1,
         b = args.b,
         compression = args.compression,
         bm25_retrieval = args.bm25)
  if args.scoring in ['all', 'lsi']:
    eval(qrels,
         k = args.k,
         scoring = 'lsi',
         compression = args.compression,
         bm25_retrieval = args.bm25,
         lsi_components = args.lsi_components,
         lsi_n_iter = args.lsi_n_iter,
         rebuild_lsi = args.rebuild_lsi)