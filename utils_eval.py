from collections import Counter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def get_ngrams(n, text):
    return Counter(t for t in zip(*([text[i:] for i in range(n)])))


def calc_rouge(hypothesis, references, n=1):
    refs_counts = Counter()
    for ref in references:
        refs_counts |= get_ngrams(n, ref)

    candidate_counts = get_ngrams(n, hypothesis)

    overlap = candidate_counts & refs_counts

    precision = sum(overlap.values()) / sum(candidate_counts.values()) if candidate_counts.values() else 0
    recall = sum(overlap.values()) / sum(refs_counts.values()) if refs_counts.values() else 0

    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    rouge_score = {'f': f1_score, 'p': precision, 'r': recall}
    return rouge_score


def get_rouge(references, hypotheses):
    assert len(references) == len(hypotheses), "Number of references and hypotheses should be same."
    rouge_N = 0.
    for n in range(1, 5):
        rouge_n = 0.
        for ref, hyp in zip(references, hypotheses):
            rouge = calc_rouge(hyp, ref, n)
            rouge_n += rouge['f']
        rouge_n = rouge_n / len(hypotheses)  # 计算每个批次平均 n-F1
        # print(f"ROUGE-{n}:", rouge_n)
        rouge_N += rouge_n
    rouge_N = rouge_N / 4.0
    return rouge_N


def get_bleu(references, hypotheses):
    assert len(references) == len(hypotheses), "Number of references and hypotheses should be same."
    hypotheses = [[str(word) for word in sent] for sent in hypotheses]
    references = [[[str(word) for word in sent] for sent in ref] for ref in references]

    # 计算 BLEU-1 至 BLEU-4 分数
    smoothie = SmoothingFunction().method4
    bleu_scores_corpus = []
    average_bleu_score = 0
    for n in range(1, 5):
        weights = [1 / n] * n + [0] * (4 - n)  # 这将会得到 (1/n, ..., 1/n, 0, ..., 0)
        bleu_score = corpus_bleu(references, hypotheses, weights=weights, smoothing_function=smoothie)
        bleu_scores_corpus.append(bleu_score)
    # 输出 corpus BLEU 结果
    for n, score in enumerate(bleu_scores_corpus, start=1):
        # print(f"Corpus BLEU-{n} Score: {score * 100:.2f}%")  # 将分数转为保留两位小数的形式
        average_bleu_score += score
    average_bleu_score /= 4.0
    # print(f"Corpus BLEU- Score: {average_bleu_score * 100:.2f}%")  # 将分数转为保留两位小数的形式
    return average_bleu_score


hypotheses = [[99, 404, 438, 438, 438, 1606, 2709, 2709, 1606, 1606], [2011, 159, 1236, 1236, 1236, 1236, 1236]]
references = [[[99, 404, 765, 4, 399, 370, 12, 2767], [692, 170, 137, 400, 287, 65, 1649, 370, 22, 12, 2767],
               [2765, 36, 12, 128, 33, 500, 220, 402, 130, 2767],
               [1, 144, 1746, 10, 4, 400, 16, 370, 4, 2357, 11, 12, 2767], [1, 404, 162, 163, 12, 3, 4, 147, 2767]],
              [[1, 25, 26, 136, 137, 139, 2767], [1, 293, 26, 7, 136, 137, 139, 2767],
               [1, 25, 26, 7, 136, 137, 139, 2767], [1, 293, 25, 26, 124, 820, 137, 2767],
               [1, 26, 7, 136, 22, 139, 202, 125, 1452, 810, 4, 681, 27, 2765, 2767]]]

if __name__ == '__main__':
    r = get_rouge(references, hypotheses)
    print(r)
    b = get_bleu(references, hypotheses)

    print(f"Corpus Rouge- Score: {r * 100:.2f}%")  # 将分数转为保留两位小数的形式
    print(f"Corpus BLEU- Score: {b * 100:.2f}%")  # 将分数转为保留两位小数的形式
