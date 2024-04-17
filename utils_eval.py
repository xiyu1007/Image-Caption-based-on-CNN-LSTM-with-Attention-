import math

import torch


def get_BLEU(caption, references, n=4):
    caption_tokens = caption.split()
    if len(caption_tokens) == 0:
        return 0
    reference_tokens = [reference.split() for reference in references]

    smoothie = 1e-10  # Smoothing value
    # weights = [0.25] * 4  # BLEU-4 weights
    weights = 0.25  # 1/N BLEU-4 weights

    # Compute n-grams for caption and references
    caption_ngrams = {k: {tuple(caption_tokens[i:i + k]) for i in range(len(caption_tokens) - k + 1)} for k in
                      range(1, n + 1)}
    # {1: {('is',), ('a',), ('cat',)}, 2: {('cat', 'is'), ('a', 'cat')}, 3: {('a', 'cat', 'is')}, 4: set()}

    reference_ngrams = [
        {k: {tuple(reference_tokens[j][i:i + k]) for i in range(len(reference_tokens[j]) - k + 1)} for k in
         range(1, n + 1)} for j in range(len(reference_tokens))]

    clipped_counts = torch.zeros(4)
    reference_len_min = float(min([len(reference) for reference in reference_tokens]))

    for k in range(n):
        caption_ngram_set = caption_ngrams[k + 1]
        max_match_count = 0
        for reference_ngram_set in reference_ngrams:
            intersection = len(caption_ngram_set & reference_ngram_set[k + 1])
            max_match_count = max(max_match_count, intersection)
        clipped_counts[k] += max_match_count

    precision = clipped_counts / len(caption_tokens)
    brevity_penalty = min(1.0, math.exp(1 - (reference_len_min / float(len(caption_tokens)))))
    bleu = sum(math.exp(weights * math.log(max(p, smoothie))) for p in precision)
    bleu *= brevity_penalty

    return bleu


def get_ROUGE_N(caption, references, n=4):
    # weights = [0.25] * 4  # BLEU-4 weights
    weights = 0.25  # 1/N BLEU-4 weights

    caption_tokens = caption.split()
    if len(caption_tokens) == 0:
        return 0
    reference_tokens = [reference.split() for reference in references]

    # Compute n-grams for caption and references
    caption_ngrams = {k: {tuple(caption_tokens[i:i + k]) for i in range(len(caption_tokens) - k + 1)} for k in
                      range(1, n + 1)}
    # {1: {('is',), ('a',), ('cat',)}, 2: {('cat', 'is'), ('a', 'cat')}, 3: {('a', 'cat', 'is')}, 4: set()}

    reference_ngrams = [
        {k: {tuple(reference_tokens[j][i:i + k]) for i in range(len(reference_tokens[j]) - k + 1)} for k in
         range(1, n + 1)} for j in range(len(reference_tokens))]

    clipped_counts = []
    reference_len = []

    for reference_ngram_set in reference_ngrams:
        intersection = []
        re_len = []
        for k in range(n):
            caption_ngram_set = caption_ngrams[k + 1]
            re_len.append(len(reference_ngram_set[k + 1]))
            intersection.append(len(caption_ngram_set & reference_ngram_set[k + 1]))
        clipped_counts.append(intersection)
        reference_len.append(re_len)

    result = [[float(clipped_counts[i][j]) / reference_len[i][j] for j in range(len(clipped_counts[i]))] for i in
              range(len(clipped_counts))]

    # recall = sum(re for re in result)/n/float(len(reference_len))
    # 计算平均召回率，这里假设 n 是总的条目数
    recall = sum(sum(re for re in row) for row in result) / n / float(len(reference_len))

    return recall


if __name__ == '__main__':
    # 示例用法
    caption = "a cat is sitting on a table"
    references = ["a cat sits on a table", "the cat is sitting on the table"]
    bleu_score = get_BLEU(caption, references)
    rouge_score = get_ROUGE_N(caption, references, n=4)

    print("BLEU Score:", bleu_score)
    print("ROUGE-1 Score:", rouge_score)
