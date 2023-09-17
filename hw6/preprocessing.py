from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    # task A
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.
    Args:
        filename: Name of the file containing XML markup for labeled alignments
    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    def alignment_help(y):
        arg = []
        if y.text is None:
            return []
        for i in y.text.split(' '):
            arg.append(tuple(int(j) for j in i.split('-')))
        return arg

    file = open(filename, 'r')
    file_str = file.read().replace('&', '&#038;')
    xml_parsing = ET.fromstring(file_str)
    sentence_pairs = []
    alignments = []
    for x in xml_parsing:
        new_sentence_pairs = SentencePair(x[0].text.split(' '), x[1].text.split(' '))
        sentence_pairs.append(new_sentence_pairs)
        arg_1 = alignment_help(x[2])
        arg_2 = alignment_help(x[3])
        new_alignments = LabeledAlignment(*[arg_1, arg_2])
        alignments.append(new_alignments)
    return sentence_pairs, alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    # C
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.
    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff -- natural number -- most frequent tokens in each language
    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    Tip:
        Use cutting by freq_cutoff independently in src and target.
        Moreover, in both cases of freq_cutoff (None or not None) - you may get a different size of the dictionary
    """
    def get_token_help(d):
        d = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
        d_list = list(d.keys())
        if freq_cutoff is not None:
            d_list = d_list[:freq_cutoff]
        res_dict = dict()
        for j in range(len(d_list)):
            res_dict[d_list[j]] = j
        return res_dict

    source_dict = dict()
    target_dict = dict()
    for sentence in sentence_pairs:
        for i in sentence.source:
            source_dict[i] = 0
        for i in sentence.target:
            target_dict[i] = 0
    for sentence in sentence_pairs:
        for i in sentence.source:
            source_dict[i] += 1
        for i in sentence.target:
            target_dict[i] += 1
    return get_token_help(source_dict), get_token_help(target_dict)


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    # C
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.

    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language
    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []
    for sentence in sentence_pairs:
        indexes_source = []
        indexes_target = []
        flag_1 = False
        flag_2 = False
        for x in sentence.source:
            if x in source_dict:
                indexes_source.append(source_dict[x])
                flag_1 = True
        for x in sentence.target:
            if x in target_dict:
                indexes_target.append(target_dict[x])
                flag_2 = True
        indexes_source = np.array(indexes_source)
        indexes_target = np.array(indexes_target)
        if flag_1 and flag_2:
            tokenized_sentence_pairs.append(TokenizedSentencePair(indexes_source, indexes_target))
    return tokenized_sentence_pairs
