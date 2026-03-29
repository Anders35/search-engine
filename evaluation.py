import argparse
import re
import math
from bsbi import BSBIIndex, get_postings_encoding

######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking)):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score


def dcg(ranking):
  """ menghitung search effectiveness metric score dengan
      Discounted Cumulative Gain (DCG)

      Parameters
      ----------
      ranking: List[int]
         vektor biner relevansi dokumen pada ranking

      Returns
      -------
      Float
        score DCG
  """
  score = 0.
  for i, rel in enumerate(ranking, start = 1):
    if i == 1:
      score += rel
    else:
      score += rel / math.log2(i + 1)
  return score


def ndcg(ranking, num_relevant = None):
  """ menghitung search effectiveness metric score dengan
      Normalized Discounted Cumulative Gain (NDCG)

      Parameters
      ----------
      ranking: List[int]
         vektor biner relevansi dokumen pada ranking
      num_relevant: int atau None
         jumlah total dokumen relevan untuk query.
         jika None, dihitung dari ranking.

      Returns
      -------
      Float
        score NDCG
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
  """ menghitung search effectiveness metric score dengan
      Average Precision (AP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner relevansi dokumen pada ranking
      num_relevant: int
         jumlah total dokumen relevan untuk query

      Returns
      -------
      Float
        score AP
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


######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

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

######## >>>>> EVALUASI !

def eval(qrels, query_file = "queries.txt", k = 1000, scoring = 'tfidf', k1 = 1.2, b = 0.75,
         compression = 'elias-gamma'):
  """ 
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
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

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      ranking = []
      for (score, doc) in BSBI_instance.retrieve(query, k = k, scoring = scoring, k1 = k1, b = b):
        did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
        ranking.append(qrels[qid][did])
      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking))
      ndcg_scores.append(ndcg(ranking, num_relevant = num_relevant))
      ap_scores.append(ap(ranking, num_relevant = num_relevant))

  print(f"Hasil evaluasi {scoring.upper()} terhadap 30 queries")
  print("RBP score =", sum(rbp_scores) / len(rbp_scores))
  print("DCG score =", sum(dcg_scores) / len(dcg_scores))
  print("NDCG score =", sum(ndcg_scores) / len(ndcg_scores))
  print("AP score =", sum(ap_scores) / len(ap_scores))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluasi retrieval dengan RBP')
  parser.add_argument('--compression', default='elias-gamma',
                      choices=['standard', 'vbe', 'elias-gamma'],
                      help='Jenis kompresi postings yang dipakai saat membaca index')
  parser.add_argument('--scoring', default='all', choices=['all', 'tfidf', 'bm25'],
                      help='Skema scoring yang dievaluasi')
  parser.add_argument('-k', type=int, default=1000, help='Top-K dokumen per query untuk evaluasi')
  parser.add_argument('--k1', type=float, default=1.2, help='Parameter BM25 k1')
  parser.add_argument('--b', type=float, default=0.75, help='Parameter BM25 b')
  args = parser.parse_args()

  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  if args.scoring in ['all', 'tfidf']:
    eval(qrels, k = args.k, scoring = 'tfidf', compression = args.compression)
  if args.scoring in ['all', 'bm25']:
    eval(qrels, k = args.k, scoring = 'bm25', k1 = args.k1, b = args.b, compression = args.compression)