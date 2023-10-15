"""
COMS W4705 - Natural Language Processing - Fall 2023
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg


### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) == 2 and \
                isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write(
                    "Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str):  # Leaf nodes may be strings
                continue
            if not isinstance(bps, tuple):
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(
                        bps))
                return False
            if len(bps) != 2:
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(
                        bps))
                return False
            for bp in bps:
                if not isinstance(bp, tuple) or len(bp) != 3:
                    sys.stderr.write(
                        "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(
                            bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write(
                        "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(
                            bp))
                    return False
    return True


def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) == 2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write(
                    "Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True


class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar):
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self, tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        n = len(tokens)
        dp_table = {}

        # Initialization
        for i in range(n):
            span = (i, i + 1)
            current = tokens[i],  # turned into a tuple
            rules = self.grammar.rhs_to_rules[current]
            if len(rules) == 0:
                dp_table[span] = dict()
            else:
                bp_dict = dict()
                for lhs, rhs, prob in rules:
                    bp_dict[lhs] = rhs[0]
                dp_table[span] = bp_dict

        # Fill up the prob table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                bp_dict = defaultdict()
                for k in range(i + 1, j):
                    B_dict = dp_table[(i, k)]
                    C_dict = dp_table[(k, j)]
                    if len(B_dict) == 0 or len(C_dict) == 0:
                        dp_table[(i, j)] = dict()
                        continue
                    for b_key, b_val in B_dict.items():
                        for c_key, c_val in C_dict.items():
                            rules = self.grammar.rhs_to_rules[(b_key, c_key)]
                            if len(rules) != 0:
                                for lhs, rhs, prob in rules:
                                    bp_dict[lhs] = ((b_key, i, k), (c_key, k, j))
                dp_table[(i, j)] = bp_dict

        # Check if the string is in the language
        if self.grammar.startsymbol in dp_table[(0, n)]:
            return True

        return False

    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        parse_table = {}
        probs_table = {}
        n = len(tokens)

        # format of the tables
        # parse_table[(0, 0)] = {"FLIGHTS": "hello", "NP": "hello"}
        # parse_table[(1, 0)] = {"PP": (("NP", 0, 2), ("FLIGHTS", 2, 3)), "NP": (("NP", 0, 2), ("FLIGHTS", 2, 3))}
        # probs_table[(0, 0)] = {"PP": -1.0, "NP": -1.5}

        # Initialization
        for i in range(n):
            span = (i, i + 1)
            current = tokens[i],  # turned into a tuple
            rules = self.grammar.rhs_to_rules[current]
            if len(rules) == 0:  # this terminal is not associated with any rules
                probs_table[span] = dict()
                parse_table[span] = dict()
            else:  # this terminal is associated with some rules
                bp_dict = dict()
                probs_dict = dict()
                rule_lhs_set = set()
                for lhs, rhs, prob in rules:
                    if lhs not in rule_lhs_set:
                        bp_dict[lhs] = rhs[0]
                        probs_dict[lhs] = math.log(prob)
                        rule_lhs_set.add(lhs)
                    else:  # choose the most likely non-terminal
                        if prob > probs_dict[lhs]:
                            bp_dict[lhs] = rhs[0]
                            probs_dict[lhs] = math.log(prob)
                parse_table[span] = bp_dict
                probs_table[span] = probs_dict

        # Fill up the prob table
        for length in range(2, n + 1):  # 2 3 4 5 6
            for i in range(n - length + 1):  # 0 1 2 3 4
                j = i + length  # 2 3 4 5 6
                bp_dict = defaultdict()
                probs_dict = defaultdict()
                nonterminal_set = set()
                for k in range(i + 1, j):
                    B_dict = parse_table[(i, k)]
                    C_dict = parse_table[(k, j)]
                    B_prob = probs_table[(i, k)]
                    C_prob = probs_table[(k, j)]
                    if len(B_dict) == 0 or len(C_dict) == 0:
                        parse_table[(i, j)] = dict()
                        continue
                    for b_key, b_val in B_dict.items():
                        for c_key, c_val in C_dict.items():
                            rules = self.grammar.rhs_to_rules[(b_key, c_key)]
                            if len(rules) != 0:
                                for lhs, rhs, prob in rules:
                                    if lhs not in nonterminal_set:
                                        bp_dict[lhs] = ((b_key, i, k), (c_key, k, j))
                                        probs_dict[lhs] = math.log(prob) + B_prob[b_key] + C_prob[c_key]
                                        nonterminal_set.add(lhs)
                                    else:
                                        new_prob = math.log(prob) + B_prob[b_key] + C_prob[c_key]
                                        if new_prob > probs_dict[lhs]:
                                            bp_dict[lhs] = ((b_key, i, k), (c_key, k, j))
                                            probs_dict[lhs] = new_prob
                parse_table[(i, j)] = bp_dict
                probs_table[(i, j)] = probs_dict

        return dict(parse_table), dict(probs_table)


def get_tree_helper(chart, i, j, root):
    if i == j or type(chart[(i, j)][root]) == str:
        ter = chart[(i, j)][root]
        return root, ter

    current = chart[(i, j)][root]
    left = get_tree_helper(chart, current[0][1], current[0][2], current[0][0])
    right = get_tree_helper(chart, current[1][1], current[1][2], current[1][0], )
    return root, left, right


def get_tree(chart, i, j, nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    return get_tree_helper(chart, i, j, nt)


if __name__ == "__main__":
    with open('atis3.pcfg', 'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)
        # toks = ['flights', 'from', 'miami', 'to', 'cleveland', '.']
        # print(parser.is_in_language(toks))
        toks = ['with', 'the', 'least', 'expensive', 'fare', '.']
        table, probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)

    # # TODO: test with lecture example, remove it
    # with open('test_rules.pcfg', 'r') as test_file:
    #     grammar = Pcfg(test_file)
    #     parser = CkyParser(grammar)
    #     toks = ['she', 'saw', 'the', 'cat', 'with', 'glasses']
    #     print(parser.is_in_language(toks))
    #     toks = ['she', 'the', 'cat', 'with', 'glasses']
    #     print(parser.is_in_language(toks))

    # # TODO: test part 4
    # with open('atis3.pcfg', 'r') as grammar_file:
    #     grammar = Pcfg(grammar_file)
    #     parser = CkyParser(grammar)
    #     toks = ['flights', 'from', 'miami', 'to', 'cleveland', '.']
    #     table, probs = parser.parse_with_backpointers(toks)
    #     assert check_table_format(table)
    #     assert check_probs_format(probs)
    #     print(get_tree(table, 0, len(toks), grammar.startsymbol))
