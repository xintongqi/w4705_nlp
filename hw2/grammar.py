"""
COMS W4705 - Natural Language Processing - Fall 2023
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum, isclose

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        lhs_set = set(self.lhs_to_rules.keys())

        for lhs in lhs_set:
            # verify prob sums to 1
            rules = self.lhs_to_rules[lhs]
            sum_of_rules = fsum(x[2] for x in rules)
            isclosed = isclose(1.0, sum_of_rules, rel_tol=1e-09, abs_tol=0.0)
            if not isclosed:
                return False

            # verify valid CNF
            for rule in rules:
                lhs, rhs, prob = rule
                if len(rhs) == 1:  # lhs -> terminal
                    if rhs[0] in lhs_set:
                        return False
                elif len(rhs) == 2:  # lhs -> non-terminal, non-terminal
                    if rhs[0] not in lhs_set or rhs[1] not in lhs_set:
                        return False
                else:
                    return False

        return True


if __name__ == "__main__":
    # with open(sys.argv[1],'r') as grammar_file:
    with open("test_rules.pcfg",'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        if grammar.verify_grammar():
            print("The grammar is a valid PCFG in CNF.")
        else:
            print("Error: the grammar is not a valid PCFG in CNF.")
        
