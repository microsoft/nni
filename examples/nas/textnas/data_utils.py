# -*- coding: UTF-8 -*-
# Taken from Jonathan K's Berkeley parser with minor modification
import sys

word_to_word_mapping = {
    '{': '-LCB-',
    '}': '-RCB-'
}
word_to_POS_mapping = {
    '--': ':',
    '-': ':',
    ';': ':',
    ':': ':',
    '-LRB-': '-LRB-',
    '-RRB-': '-RRB-',
    '-LCB-': '-LRB-',
    '-RCB-': '-RRB-',
    '{': '-LRB-',
    '}': '-RRB-',
    'Wa': 'NNP'
}


def standardise_node(tree):
    if tree.word in word_to_word_mapping:
        tree.word = word_to_word_mapping[tree.word]

    # HOS: don't wanna change labels for now (fix later)
    #if tree.word in word_to_POS_mapping:
    #    tree.label = word_to_POS_mapping[tree.word]


class PTB_Tree:
    '''Tree for PTB format

	>>> tree = PTB_Tree()
	>>> tree.set_by_text("(ROOT (NP (NNP Newspaper)))")
	>>> print tree
	(ROOT (NP (NNP Newspaper)))
	>>> tree = PTB_Tree()
	>>> tree.set_by_text("(ROOT (S (NP-SBJ (NNP Ms.) (NNP Haag) ) (VP (VBZ plays) (NP (NNP Elianti) )) (. .) ))")
	>>> print tree
	(ROOT (S (NP-SBJ (NNP Ms.) (NNP Haag)) (VP (VBZ plays) (NP (NNP Elianti))) (. .)))
	>>> print tree.word_yield()
	Ms. Haag plays Elianti .
	>>> tree = PTB_Tree()
	>>> tree.set_by_text("(ROOT (NFP ...))")
	>>> print tree
	(ROOT (NFP ...))
	>>> tree.word_yield()
	'...'
	'''
    # Convert text from the PTB to a tree. For example:
    # ( (S (NP-SBJ (NNP Ms.) (NNP Haag) ) (VP (VBZ plays) (NP (NNP Elianti) )) (. .) ))
    # This is a compressed form of:
    # ( (S
    # (NP-SBJ (NNP Ms.) (NNP Haag))
    #     (VP (VBZ plays)
    #       (NP (NNP Elianti)))
    #     (. .)))
    def __init__(self):
        self.subtrees = []
        self.word = None
        self.label = ''
        self.parent = None
        self.span = (-1, -1)
        self.word_vector = None # HOS, store dx1 RNN word vector
        self.prediction = None # HOS, store Kx1 prediction vector

    def is_leaf(self):
        return len(self.subtrees) == 0

    def set_by_text(self, text, pos=0, left=0):
        depth = 0
        right = left
        for i in range(pos + 1, len(text)):
            char = text[i]
            # update the depth
            if char == '(':
                depth += 1
                if depth == 1:
                    subtree = PTB_Tree()
                    subtree.parent = self
                    subtree.set_by_text(text, i, right)
                    right = subtree.span[1]
                    self.span = (left, right)
                    self.subtrees.append(subtree)
            elif char == ')':
                depth -= 1
                if len(self.subtrees) == 0:
                    pos = i
                    for j in range(i, 0, -1):
                        if text[j] == ' ':
                            pos = j
                            break
                    self.word = text[pos + 1:i]
                    self.span = (left, left + 1)

            # we've reached the end of the category that is the root of this subtree
            if depth == 0 and char == ' ' and self.label == '':
                self.label = text[pos + 1:i]
            # we've reached the end of the scope for this bracket
            if depth < 0:
                break

        # Fix some issues with variation in output, and one error in the treebank
        # for a word with a punctuation POS
        standardise_node(self)

    def clone(self):
        ans = PTB_Tree()
        ans.word = self.word
        ans.label = self.label
        ans.parent = None
        ans.span = self.span
        ans.subtrees = []
        for subtree in self.subtrees:
            ans.subtrees.append(subtree.clone())
            ans.subtrees[-1].parent = ans
        return ans

    def word_yield(self, span=None, pos=-1):
        return_tuple = True
        if pos < 0:
            pos = 0
            return_tuple = False
        ans = None
        if self.word is not None:
            if span is None or span[0] <= pos < span[1]:
                ans = (pos + 1, self.word)
            else:
                ans = (pos + 1, '')
        else:
            text = []
            for subtree in self.subtrees:
                pos, words = subtree.word_yield(span, pos)
                if words != '':
                    text.append(words)
            ans = (pos, ' '.join(text))
        if return_tuple:
            return ans
        else:
            return ans[1]

    def __repr__(self, single_line=True, depth=0):
        ans = ''
        if not single_line and depth > 0:
            ans = '\n' + depth * '\t'
        ans += '(' + self.label
        if self.word is not None:
            ans += ' ' + self.word
        for subtree in self.subtrees:
            if single_line:
                ans += ' '
            ans += subtree.__repr__(single_line, depth + 1)
        ans += ')'
        return ans

    def calculate_spans(self, left=0):
        right = left
        if self.word is not None:
            right += 1
        else:
            for subtree in self.subtrees:
                right = subtree.calculate_spans(right)
        self.span = (left, right)
        return right

    def check_consistency(self):
        if len(self.subtrees) > 0:
            for subtree in self.subtrees:
                if subtree.parent != self:
                    print("bad parent link")
                    print(id(self), id(subtree.parent), id(subtree))
                    ###					print self
                    print(subtree)
                    ###					print subtree.parent
                    return False
                if not subtree.check_consistency():
                    return False
            if self.span[0] != self.subtrees[0].span[0]:
                print("incorrect span")
                return False
            if self.span[1] != self.subtrees[-1].span[1]:
                print("incorrect span")
                return False
        return True

    def span_list(self, span_list=None):
        if span_list is None:
            span_list = []
        for subtree in self.subtrees:
            subtree.span_list(span_list)
        span_list.append((self.span[0], self.span[1], self))
        return span_list

    def get_lowest_span(self, start=-1, end=-1):
        if start <= end < 0:
            return None
        # TODO: Optimise this loop to prevent unnecessary recursion
        for subtree in self.subtrees:
            ans = subtree.get_lowest_span(start, end)
            if ans is not None:
                return ans
        if self.span[1] == end or end < 0:
            if self.span[0] == start or start < 0:
                return self
        return None

    def get_highest_span(self, start=-1, end=-1):
        if start <= end < 0:
            return None
        # TODO: Optimise this loop to prevent unnecessary recursion
        if self.span[1] == end or end < 0:
            if self.span[0] == start or start < 0:
                return self
        for subtree in self.subtrees:
            ans = subtree.get_highest_span(start, end)
            if ans is not None:
                return ans
        return None

    def get_spans(self, start=-1, end=-1, span_list=None):
        if start <= end < 0:
            return self.span_list()
        if span_list is None:
            span_list = []
        # TODO: Optimise this loop to prevent unnecessary recursion
        for subtree in self.subtrees:
            subtree.get_spans(start, end, span_list)
        if self.span[1] == end or end < 0:
            if self.span[0] == start or start < 0:
                span_list.append((self.span[0], self.span[1], self))
        return span_list

    def get_errors(self, gold):
        ans = error_set.Error_Set()
        gold_spans = gold.span_list()
        test_spans = self.span_list()
        gold_spans.sort()
        test_spans.sort()
        test_span_set = {}
        for span in test_spans:
            key = (span[0], span[1], span[2].label)
            if key not in test_span_set:
                test_span_set[key] = 0
            test_span_set[key] += 1
        gold_span_set = {}
        for span in gold_spans:
            key = (span[0], span[1], span[2].label)
            if key not in gold_span_set:
                gold_span_set[key] = 0
            gold_span_set[key] += 1

        # Extra
        for span in test_spans:
            key = (span[0], span[1], span[2].label)
            if key not in gold_span_set or gold_span_set[key] < 1:
                if span[2].word is None:
                    ans.add_error('extra', (span[0], span[1]), span[2].label, span[2])
            else:
                gold_span_set[key] -= 1

        # Missing and crossing
        for span in gold_spans:
            key = (span[0], span[1], span[2].label)
            if key not in test_span_set or test_span_set[key] < 1:
                if span[2].word is not None:
                    continue
                is_crossing = False
                for tspan in test_span_set:
                    if tspan[0] < span[0] < tspan[1] < span[1]:
                        is_crossing = True
                        break
                    if span[0] < tspan[0] < span[1] < tspan[1]:
                        is_crossing = True
                        break
                if is_crossing:
                    ans.add_error('crossing', (span[0], span[1]), span[2].label, span[2])
                else:
                    ans.add_error('missing', (span[0], span[1]), span[2].label, span[2])
            else:
                test_span_set[key] -= 1
        return ans

    def colour_repr(self, gold=None, depth=0, single_line=False, missing=None, extra=None):
        '''Pretty print, with errors marked using colour.

		'missing' should contain tuples:
			(start, end, label, crossing-T/F)
		'''
        if missing is None:
            if gold is None:
                return "Error - no gold tree and no missing list for colour repr"
            # look at gold and work out what missing should be
            errors = self.get_errors(gold)
            extra = [e[3] for e in errors.extra]
            extra = set(extra)
            missing = [(e[1][0], e[1][1], e[2], False) for e in errors.missing]
            missing += [(e[1][0], e[1][1], e[2], True) for e in errors.crossing]
        start_missing = "\033[01;36m"
        start_extra = "\033[01;31m"
        start_crossing = "\033[01;33m"
        end_colour = "\033[00m"
        ans = ''
        if not single_line:
            ans += '\n' + depth * '\t'

        # start of this
        if self in extra:
            ans += start_extra + '(' + self.label + end_colour
        else:
            ans += '(' + self.label

        # crossing brackets starting
        if self.parent is None or self.parent.subtrees[0] != self:
            # these are marked as high as possible
            labels = []
            for error in missing:
                if error[0] == self.span[0] and error[3]:
                    labels.append((error[1], error[2]))
            labels.sort(reverse=True)
            if len(labels) > 0:
                ans += ' ' + start_crossing + ' '.join(['(' + label[1] for label in labels]) + end_colour

        # word
        if self.word is not None:
            ans += ' ' + self.word

        # subtrees
        below = []
        for subtree in self.subtrees:
            text = subtree.colour_repr(gold, depth + 1, single_line, missing, extra)
            if single_line:
                text = ' ' + text
            below.append([subtree.span[0], subtree.span[1], text])
        # add missing brackets that surround subtrees
        for length in range(1, len(below)):
            for i in range(len(below)):
                j = i + length
                if i == 0 and j == len(below) - 1:
                    continue
                if j >= len(below):
                    continue
                for error in missing:
                    if below[i][0] == error[0] and below[j][1] == error[1] and not error[3]:
                        start = below[i][2].split('(')[0]
                        for k in range(i, j + 1):
                            below[k][2] = '\n\t'.join(below[k][2].split('\n'))
                        below[i][2] = start + start_missing + '(' + error[2] + end_colour + below[i][2]
                        below[j][2] += start_missing + ')' + end_colour
        ans += ''.join([part[2] for part in below])

        # end of this
        if self in extra:
            ans += start_extra + ')' + end_colour
        else:
            ans += ')'

        if self.parent is None or self.parent.subtrees[-1] != self:
            # if there are crossing brackets that end here, mark that
            labels = []
            for error in missing:
                if error[1] == self.span[1] and error[3]:
                    labels.append((error[0], error[2]))
            labels.sort(reverse=True)
            if len(labels) > 0:
                ans += ' ' + start_crossing + ' '.join([label[1] + ')' for label in labels]) + end_colour

        if self.parent is None or len(self.parent.subtrees) > 1:
            # check for missing brackets that go around this node
            for error in missing:
                if error[0] == self.span[0] and error[1] == self.span[1] and not error[3]:
                    if not self in extra:
                        # Put them on a new level
                        ans = '\n\t'.join(ans.split('\n'))
                        extra_text = '\n' + depth * '\t'
                        extra_text += start_missing + '(' + error[2] + end_colour
                        ans = extra_text + ans
                        ans += start_missing + ')' + end_colour
                    else:
                        # Put them on the same line
                        start = 0
                        for char in ans:
                            if char not in '\n\t':
                                break
                            start += 1
                        pretext = ans[:start]
                        ans = ans[start:]
                        extra_text = start_missing + '(' + error[2] + end_colour + ' '
                        ans = pretext + extra_text + ans
                        ans += start_missing + ')' + end_colour
        return ans


def remove_traces(tree, left=0):
    '''Adjust the tree to remove traces
	>>> tree = PTB_Tree()
	>>> tree.set_by_text("(ROOT (S (PP (IN By) (NP (CD 1997))) (, ,) (NP (NP (ADJP (RB almost) (DT all)) (VBG remaining) (NNS uses)) (PP (IN of) (NP (JJ cancer-causing) (NN asbestos)))) (VP (MD will) (VP (VB be) (VP (VBN outlawed) (NP (-NONE- *-6))))) (. .)))")
	>>> ctree = remove_traces(tree)
	>>> print ctree
	(ROOT (S (PP (IN By) (NP (CD 1997))) (, ,) (NP (NP (ADJP (RB almost) (DT all)) (VBG remaining) (NNS uses)) (PP (IN of) (NP (JJ cancer-causing) (NN asbestos)))) (VP (MD will) (VP (VB be) (VP (VBN outlawed)))) (. .)))
	'''
    if tree.label == '-NONE-':
        return None
    right = left
    if tree.word is not None:
        right = left + 1
    subtrees = []
    for subtree in tree.subtrees:
        nsubtree = remove_traces(subtree, right)
        if nsubtree != None:
            subtrees.append(nsubtree)
            right = nsubtree.span[1]
    if tree.word is None and len(subtrees) == 0:
        return None
    ans = PTB_Tree()
    ans.word = tree.word
    ans.label = tree.label
    ans.span = (left, right)
    ans.subtrees = subtrees
    for subtree in subtrees:
        subtree.parent = ans
    return ans


def remove_function_tags(tree):
    '''Adjust the tree to remove function tags on labels
	>>> tree = PTB_Tree()
	>>> tree.set_by_text("(ROOT (S (NP-SBJ (NNP Ms.) (NNP Haag)) (VP (VBZ plays) (NP (NNP Elianti))) (. .)))")
	>>> ctree = remove_function_tags(tree)
	>>> print ctree
	(ROOT (S (NP (NNP Ms.) (NNP Haag)) (VP (VBZ plays) (NP (NNP Elianti))) (. .)))

	# don't remove brackets
	>>> tree = PTB_Tree()
	>>> tree.set_by_text("(ROOT (S (NP-SBJ (`` ``) (NP-TTL (NNP Funny) (NNP Business)) ('' '') (PRN (-LRB- -LRB-) (NP (NNP Soho)) (, ,) (NP (CD 228) (NNS pages)) (, ,) (NP ($ $) (CD 17.95)) (-RRB- -RRB-)) (PP (IN by) (NP (NNP Gary) (NNP Katzenstein)))) (VP (VBZ is) (NP-PRD (NP (NN anything)) (PP (RB but)))) (. .)))")
	>>> ctree = remove_function_tags(tree)
	>>> print ctree
	(ROOT (S (NP (`` ``) (NP (NNP Funny) (NNP Business)) ('' '') (PRN (-LRB- -LRB-) (NP (NNP Soho)) (, ,) (NP (CD 228) (NNS pages)) (, ,) (NP ($ $) (CD 17.95)) (-RRB- -RRB-)) (PP (IN by) (NP (NNP Gary) (NNP Katzenstein)))) (VP (VBZ is) (NP (NP (NN anything)) (PP (RB but)))) (. .)))
	'''
    ans = PTB_Tree()
    ans.word = tree.word
    ans.label = tree.label
    if not ans.label[0] == '-':
        ans.label = ans.label.split('-')[0]
    ans.label = ans.label.split('=')[0]
    ans.span = (tree.span[0], tree.span[1])
    ans.subtrees = []
    for subtree in tree.subtrees:
        nsubtree = remove_function_tags(subtree)
        ans.subtrees.append(nsubtree)
        nsubtree.parent = ans
    return ans

# Applies rules to strip out the parts of the tree that are not used in the
# standard evalb evaluation
labels_to_ignore = set(["-NONE-"])
words_to_ignore = set(["'", "`", "''", "``", "--", ":", ";", "-", ",", ".", "...", ".", "?", "!"])
POS_to_convert = {'PRT': 'ADVP'}


def apply_collins_rules(tree, left=0):
    '''Adjust the tree to remove parts not evaluated by the standard evalb
	config.

	# cutting punctuation and -X parts of labels
	>>> tree = PTB_Tree()
	>>> tree.set_by_text("(ROOT (S (NP-SBJ (NNP Ms.) (NNP Haag) ) (VP (VBZ plays) (NP (NNP Elianti) )) (. .) ))")
	>>> ctree = apply_collins_rules(tree)
	>>> print ctree
	(ROOT (S (NP (NNP Ms.) (NNP Haag)) (VP (VBZ plays) (NP (NNP Elianti)))))
	>>> print ctree.word_yield()
	Ms. Haag plays Elianti

	# cutting nulls
	>>> tree = PTB_Tree()
	>>> tree.set_by_text("(ROOT (S (PP-TMP (IN By) (NP (CD 1997))) (, ,) (NP-SBJ-6 (NP (ADJP (RB almost) (DT all)) (VBG remaining) (NNS uses)) (PP (IN of) (NP (JJ cancer-causing) (NN asbestos)))) (VP (MD will) (VP (VB be) (VP (VBN outlawed) (NP (-NONE- *-6))))) (. .)))")
	>>> ctree = apply_collins_rules(tree)
	>>> print ctree
	(ROOT (S (PP (IN By) (NP (CD 1997))) (NP (NP (ADJP (RB almost) (DT all)) (VBG remaining) (NNS uses)) (PP (IN of) (NP (JJ cancer-causing) (NN asbestos)))) (VP (MD will) (VP (VB be) (VP (VBN outlawed))))))

	# changing PRT to ADVP
	>>> tree = PTB_Tree()
	>>> tree.set_by_text("(ROOT (S (NP-SBJ-41 (DT That) (NN fund)) (VP (VBD was) (VP (VBN put) (NP (-NONE- *-41)) (PRT (RP together)) (PP (IN by) (NP-LGS (NP (NNP Blackstone) (NNP Group)) (, ,) (NP (DT a) (NNP New) (NNP York) (NN investment) (NN bank)))))) (. .)))")
	>>> ctree = apply_collins_rules(tree)
	>>> print ctree
	(ROOT (S (NP (DT That) (NN fund)) (VP (VBD was) (VP (VBN put) (ADVP (RP together)) (PP (IN by) (NP (NP (NNP Blackstone) (NNP Group)) (NP (DT a) (NNP New) (NNP York) (NN investment) (NN bank))))))))

	# not removing brackets
	>>> tree = PTB_Tree()
	>>> tree.set_by_text("(ROOT (S (NP-SBJ (`` ``) (NP-TTL (NNP Funny) (NNP Business)) ('' '') (PRN (-LRB- -LRB-) (NP (NNP Soho)) (, ,) (NP (CD 228) (NNS pages)) (, ,) (NP ($ $) (CD 17.95) (-NONE- *U*)) (-RRB- -RRB-)) (PP (IN by) (NP (NNP Gary) (NNP Katzenstein)))) (VP (VBZ is) (NP-PRD (NP (NN anything)) (PP (RB but) (NP (-NONE- *?*))))) (. .)))")
	>>> ctree = apply_collins_rules(tree)
	>>> print ctree
	(ROOT (S (NP (NP (NNP Funny) (NNP Business)) (PRN (-LRB- -LRB-) (NP (NNP Soho)) (NP (CD 228) (NNS pages)) (NP ($ $) (CD 17.95)) (-RRB- -RRB-)) (PP (IN by) (NP (NNP Gary) (NNP Katzenstein)))) (VP (VBZ is) (NP (NP (NN anything)) (PP (RB but))))))
	'''
    if tree.label in labels_to_ignore:
        return None
    if tree.word in words_to_ignore:
        return None
    ans = PTB_Tree()
    ans.word = tree.word
    ans.label = tree.label
    ans.span = (left, -1)
    right = left
    if ans.word is not None:
        right = left + 1
        ans.span = (left, right)
    subtrees = []
    ans.subtrees = subtrees
    for subtree in tree.subtrees:
        nsubtree = apply_collins_rules(subtree, right)
        if nsubtree != None:
            subtrees.append(nsubtree)
            nsubtree.parent = ans
            right = nsubtree.span[1]
    ans.span = (left, right)
    if ans.word is None and len(ans.subtrees) == 0:
        return None
    if ans.label in POS_to_convert:
        ans.label = POS_to_convert[ans.label]
    if not ans.label[0] == '-':
        ans.label = ans.label.split('-')[0]
    ans.label = ans.label.split('=')[0]
    return ans


def read_tree(source):
    '''Read a single tree from the given file.

	>>> from StringIO import StringIO
	>>> file_text = """(ROOT (S
	...   (NP-SBJ (NNP Scotty) )
	...   (VP (VBD did) (RB not)
	...     (VP (VB go)
	...       (ADVP (RB back) )
	...       (PP (TO to)
	...         (NP (NN school) ))))
	...   (. .) ))"""
	>>> in_file = StringIO(file_text)
	>>> tree = read_tree(in_file)
	>>> print tree
	(ROOT (S (NP-SBJ (NNP Scotty)) (VP (VBD did) (RB not) (VP (VB go) (ADVP (RB back)) (PP (TO to) (NP (NN school))))) (. .)))'''
    cur_text = []
    depth = 0
    while True:
        line = source.readline()
        # Check if we are out of input
        if line == '':
            return None
        # strip whitespace and only use if this contains something
        line = line.strip()
        if line == '':
            continue
        cur_text.append(line)
        # Update depth
        for char in line:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
        # At depth 0 we have a complete tree
        if depth == 0:
            tree = PTB_Tree()
            tree.set_by_text(' '.join(cur_text))
            return tree
    return None


def read_trees(source, max_sents=-1):
    '''Read a single tree from the given file.

	>>> from StringIO import StringIO
	>>> file_text = """(ROOT (S
	...   (NP-SBJ (NNP Scotty) )
	...   (VP (VBD did) (RB not)
	...     (VP (VB go)
	...       (ADVP (RB back) )
	...       (PP (TO to)
	...         (NP (NN school) ))))
	...   (. .) ))
	...
	... (ROOT (S
	... 		(NP-SBJ (DT The) (NN bandit) )
	... 		(VP (VBZ laughs)
	... 			(PP (IN in)
	... 				(NP (PRP$ his) (NN face) )))
	... 		(. .) ))"""
	>>> in_file = StringIO(file_text)
	>>> trees = read_trees(in_file)
	>>> for tree in trees:
	...   print tree
	(ROOT (S (NP-SBJ (NNP Scotty)) (VP (VBD did) (RB not) (VP (VB go) (ADVP (RB back)) (PP (TO to) (NP (NN school))))) (. .)))
	(ROOT (S (NP-SBJ (DT The) (NN bandit)) (VP (VBZ laughs) (PP (IN in) (NP (PRP$ his) (NN face)))) (. .)))'''
    if type(source) == type(''):
        source = open(source)
    trees = []
    while True:
        tree = read_tree(source)
        if tree is None:
            break
        trees.append(tree)
        if len(trees) >= max_sents > 0:
            break
    return trees


def counts_for_prf(test, gold):
    test_spans = [span for span in test.span_list() if span[2].word is None]
    gold_spans = [span for span in gold.span_list() if span[2].word is None]
    # -1 for the top node
    test_count = len(test_spans) - 1
    gold_count = len(gold_spans) - 1
    errors = test.get_errors(gold)
    tmatch = test_count - len(errors.extra)
    gmatch = gold_count - len(errors.missing) - len(errors.crossing)
    assert tmatch == gmatch
    return tmatch, gold_count, test_count


#####################################################################
#
# Main (and related functions)
#
#####################################################################

def mprint(text, out_dict, out_name):
    all_stdout = True
    for key in out_dict:
        if out_dict[key] != sys.stdout:
            all_stdout = False

    if all_stdout:
        print (text)
    elif out_name == 'all':
        for key in out_dict:
            print >> out_dict[key], text
    else:
        print >> out_dict[out_name], text


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ("Usage:")
        print ("Print trees, each on a single line:")
        print ("   %s <filename>" % sys.argv[0])
        print ("Print trees with colours to indicate errors (red for extra, blue for missing, yellow for crossing missing)")
        print ("   %s <gold> <test> [<output_prefix> all to stdout by default]" % sys.argv[0])
        print ("Running doctest")
        import doctest

        doctest.testmod()
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        trees = read_trees(filename)
        print >> sys.stderr, len(trees), "trees read from", filename
        print >> sys.stderr, "Printing trees"
        for tree in trees:
            print (tree)
    elif len(sys.argv) >= 3:
        out = {
            'err': sys.stdout,
            'notrace': sys.stdout,
            'nofunc': sys.stdout,
            'post_collins': sys.stdout
        }
        if len(sys.argv) > 3:
            prefix = sys.argv[3]
            for key in out:
                out[key] = open(prefix + '.' + key, 'w')
        mprint("Printing trees with errors coloured", out, 'all')
        gold_in = open(sys.argv[1])
        test_in = open(sys.argv[2])
        sent_no = 0
        stats = {
            'notrace': [0, 0, 0],
            'nofunc': [0, 0, 0],
            'post_collins': [0, 0, 0]
        }
        while True:
            sent_no += 1
            gold_text = gold_in.readline()
            test_text = test_in.readline()
            if gold_text == '' and test_text == '':
                mprint("End of both input files", out, 'err')
                break
            elif gold_text == '':
                mprint("End of gold input", out, 'err')
                break
            elif test_text == '':
                mprint("End of test input", out, 'err')
                break

            mprint("Sentence %d:" % sent_no, out, 'all')

            gold_text = gold_text.strip()
            test_text = test_text.strip()
            if len(gold_text) == 0:
                mprint("No gold tree", out, 'all')
                continue
            elif len(test_text) == 0:
                mprint("Not parsed", out, 'all')
                continue

            gold_complete_tree = PTB_Tree()
            gold_complete_tree.set_by_text(gold_text)
            gold_notrace_tree = remove_traces(gold_complete_tree)
            gold_nofunc_tree = remove_function_tags(gold_notrace_tree)
            gold_tree = apply_collins_rules(gold_complete_tree)
            if gold_tree is None:
                mprint("Empty gold tree", out, 'all')
                mprint(gold_complete_tree.__repr__(), out, 'all')
                mprint(gold_tree.__repr__(), out, 'all')
                continue

            test_complete_tree = PTB_Tree()
            test_complete_tree.set_by_text(test_text)
            test_notrace_tree = remove_traces(test_complete_tree)
            test_nofunc_tree = remove_function_tags(test_notrace_tree)
            test_tree = apply_collins_rules(test_complete_tree)
            if test_tree is None:
                mprint("Empty test tree", out, 'all')
                mprint(test_complete_tree.__repr__(), out, 'all')
                mprint(test_tree.__repr__(), out, 'all')
                continue

            gold_words = gold_tree.word_yield()
            test_words = test_tree.word_yield()
            if len(test_words.split()) != len(gold_words.split()):
                mprint("Sentence lengths do not match...", out, 'all')
                mprint("Gold: " + gold_words.__repr__(), out, 'all')
                mprint("Test: " + test_words.__repr__(), out, 'all')

            mprint("After removing traces:", out, 'notrace')
            mprint(test_notrace_tree.colour_repr(gold=gold_notrace_tree).strip(), out, 'notrace')
            match, gold, test = counts_for_prf(test_notrace_tree, gold_notrace_tree)
            stats['notrace'][0] += match
            stats['notrace'][1] += gold
            stats['notrace'][2] += test
            p, r, f = util.calc_prf(match, gold, test)
            mprint("%.2f  %.2f  %.2f" % (p * 100, r * 100, f * 100), out, 'notrace')

            mprint("After removing traces and function tags:", out, 'nofunc')
            mprint(test_nofunc_tree.colour_repr(gold=gold_nofunc_tree).strip(), out, 'nofunc')
            match, gold, test = counts_for_prf(test_nofunc_tree, gold_nofunc_tree)
            stats['nofunc'][0] += match
            stats['nofunc'][1] += gold
            stats['nofunc'][2] += test
            p, r, f = util.calc_prf(match, gold, test)
            mprint("%.2f  %.2f  %.2f" % (p * 100, r * 100, f * 100), out, 'nofunc')

            mprint("After applying collins rules:", out, 'post_collins')
            mprint(test_tree.colour_repr(gold=gold_tree).strip(), out, 'post_collins')
            match, gold, test = counts_for_prf(test_tree, gold_tree)
            stats['post_collins'][0] += match
            stats['post_collins'][1] += gold
            stats['post_collins'][2] += test
            p, r, f = util.calc_prf(match, gold, test)
            mprint("%.2f  %.2f  %.2f" % (p * 100, r * 100, f * 100), out, 'post_collins')

            mprint("", out, 'all')
        for key in ['notrace', 'nofunc', 'post_collins']:
            match = stats[key][0]
            gold = stats[key][1]
            test = stats[key][2]
            p, r, f = util.calc_prf(match, gold, test)
            mprint("Overall %s: %.2f  %.2f  %.2f" % (key, p * 100, r * 100, f * 100), out, key)


# -*- coding: UTF-8 -*-
import os
import io
import sys
import csv
import json

import numpy as np
import math
import random

global only_sentence, use_embed_layer, phrase_min_length, slot_num, word_embed_model
only_sentence = False
phrase_min_length=1
embed_dim = 300
slot_num = 1
word_embed_model = {}

def sst_load_trees(filename):
  trees = read_trees(filename)
  return trees

def sst_get_id_input(content, word_id_dict, max_input_length):
  #words = content.lower().replace("-", " ").replace("\\", " ").replace("/", " ").split(' ')
  words = content.split(' ')
  sentence = []
  field, pos = [], []
  index, f_index, p_index = 0, 1, 0
  for i in range(0, max(max_input_length, len(words))):
    #while (len(words) > index and
    #    len(words[index].strip()) <= 1) :
    #  index = index + 1
    if len(words) <= index: #None
      id = word_id_dict["<pad>"]
    else:
      word = words[index].strip()
      #if word in word_embed_model and word in word_id_dict:
      if word in word_id_dict:
        id = word_id_dict[word]
      else:
        id = word_id_dict["<unknown>"]
        #print ("Error: Missing {0}".format(words[index]))
    if index < max_input_length:
      sentence.append(id)
      if len(words) <= index:
        field.append(0) #field padding
        pos.append(0)
      else:
        field.append(1) #mask
        pos.append(p_index)
        p_index += 1
    index += 1
    if "<sep>" in word_id_dict and id == word_id_dict["<sep>"]: #
      f_index += 1
      p_index = 0
      pos[-1] = 0
  return sentence, field, pos

def sst_get_phrases(trees, sample_ratio=1.0, is_binary=False, only_sentence=False):
  all_phrases = []
  for tree in trees:
    if only_sentence == True:
      sentence = get_sentence_by_tree(tree)
      label = int(tree.label)
      pair = (sentence, label)
      all_phrases.append(pair)
    else:
      phrases = get_phrases_by_tree(tree)
      sentence = get_sentence_by_tree(tree)
      pair = (sentence, int(tree.label))
      all_phrases.append(pair)
      all_phrases.extend(phrases)
  np.random.shuffle(all_phrases)
  result_phrases = []
  for pair in all_phrases:
    if is_binary:
      phrase = pair[0]
      label = pair[1]
      if label <= 1:
        pair = (phrase, 0)
      elif label >= 3:
        pair = (phrase, 1)
      else:
        continue
    if sample_ratio == 1.0:
      result_phrases.append(pair)
    else:
      rand_portion = np.random.random()
      if rand_portion < sample_ratio:
        result_phrases.append(pair)
  return result_phrases

def get_phrases_by_tree(tree):
  phrases = []
  if tree == None:
    return phrases
  if tree.is_leaf():
    pair = (tree.word, int(tree.label))
    phrases.append(pair)
    return phrases
  left_child_phrases = get_phrases_by_tree(tree.subtrees[0])
  right_child_phrases = get_phrases_by_tree(tree.subtrees[1])
  phrases.extend(left_child_phrases)
  phrases.extend(right_child_phrases)
  sentence = get_sentence_by_tree(tree)
  pair = (sentence, int(tree.label))
  phrases.append(pair)
  return phrases

def get_sentence_by_tree(tree):
  sentence = ""
  if tree == None:
    return sentence
  if tree.is_leaf():
    return tree.word
  left_sentence = get_sentence_by_tree(tree.subtrees[0])
  right_sentence = get_sentence_by_tree(tree.subtrees[1])
  sentence = left_sentence + " " + right_sentence
  return sentence.strip()

def get_word_id_dict(word_num_dict, word_id_dict, min_count):
  z = [k for k in sorted(word_num_dict.keys())]
  for word in z:
    count = word_num_dict[word]
    if count >= min_count:
      index = len(word_id_dict)
      if word not in word_id_dict:
        word_id_dict[word] = index
  return word_id_dict

def load_word_num_dict(phrases, word_num_dict):
  for (sentence, label) in phrases:
    #words = sentence.lower().replace("-", " ").replace("\\", " ").replace("/", " ").split(' ')
    words = sentence.split(' ')
    for cur_word in words:
      word = cur_word.strip()
      #if len(word) <= 1: continue
      if word not in word_num_dict:
        word_num_dict[word] = 1
      else:
        count = word_num_dict[word]
        word_num_dict[word] += 1
  return word_num_dict

def load_word_id_dict_ptb(x_vector, word_id_dict):
  print(x_vector.shape)
  for x in x_vector:
    print(x.shape)
    for cur_word in x:
      word = cur_word.strip()
      if len(word) <= 1: continue
      if word not in word_id_dict:
        index = len(word_id_dict)
        word_id_dict[word] = index
  return(word_id_dict)

def init_trainable_embedding(embedding, word_id_dict,
                word_embed_model, unknown_word_embed):
  embed_dim = unknown_word_embed.shape[0]
  embedding[0] = np.zeros(embed_dim)
  embedding[1] = unknown_word_embed
  for word in word_id_dict:
    id = word_id_dict[word]
    if id == 0 or id == 1:
      #print id, word, (word in word_embed_model)
      continue
    if word in word_embed_model:
      embedding[id] = word_embed_model[word]
    else:
      embedding[id] = np.random.rand(embed_dim) / 2.0 - 0.25
      #embedding[id] = unknown_word_embed
      #print "unknwon: {0}".format(word)
  #print("init embedding: {0}".format(embedding))
  return embedding

def sst_get_trainable_data(phrases, word_id_dict, word_embed_model,
                   split_label, max_input_length, is_binary):
  images, bow_images, labels, mask = [], [], [], []

  for (phrase, label) in phrases:
    if len(phrase.split(' ')) < phrase_min_length:
      continue
    phrase_input, field_input, pos_input = sst_get_id_input(phrase, word_id_dict, max_input_length)
    #if phrase_input[0] == 0:
    #  continue
    images.append(phrase_input)
    bow_images.append(pos_input)
    labels.append(int(label))
    mask.append(field_input) #field_input is mask
  labels = np.array(labels, dtype=np.int32)
  if split_label == 1:
    split_label_str = "train"
  elif split_label == 2:
    split_label_str = "test"
  else:
    split_label_str = "valid"
  images = np.reshape(images, [-1, max_input_length])  #(N, len)
  images = images.astype(np.int32)
  mask = np.reshape(mask, [-1, max_input_length])  #(N, len)
  mask = mask.astype(np.int32)
  bow_images = np.reshape(bow_images, [-1, max_input_length])
  bow_images = bow_images.astype(np.int32)
  print(split_label_str, images.shape, labels.shape, mask.shape)
  #print(images)
  return images, bow_images, labels, mask

def load_glove_model(filename):
  embedding_dict = {}
  with open(filename) as f:
    for line in f:
      vocab_word, vec = line.strip().split(' ', 1)
      embed_vector = list(map(float, vec.split()))
      embedding_dict[vocab_word] = embed_vector 
  return embedding_dict

def load_embedding(embedding_model):
  global word_embed_model
  
  if len(word_embed_model) == 0: #avoid secondary loading
    if embedding_model == "glove" or embedding_model == "all":
      embedding_data_path = './'
      embedding_data_file = os.path.join(embedding_data_path, 'glove.840B.300d.txt')
      word_embed_model["glove"] = load_glove_model(embedding_data_file)

  unknown_word_embed = np.random.rand(embed_dim)
  unknown_word_embed = (unknown_word_embed - 0.5) / 2.0
  return word_embed_model, unknown_word_embed

def read_data_sst(word_id_dict, word_num_dict, data_path, max_input_length, embedding_model, min_count,
                  train_ratio, valid_ratio, is_binary=False, is_valid=False, cache={}):
  """Reads SST format data. Always returns NHWC format

  Returns:
    sentences: np tensor of size [N, H, W, C=1]
    labels: np tensor of size [N]
  """

  images, labels, mask = {}, {}, {}

  if len(cache) == 0:
    print("-" * 80)
    print("Reading SST data")

    train_file_name = os.path.join(data_path, 'train.txt')
    valid_file_name = os.path.join(data_path, 'dev.txt')
    test_file_name = os.path.join(data_path, 'test.txt')

    train_trees = sst_load_trees(train_file_name)
    train_phrases = sst_get_phrases(train_trees, train_ratio, is_binary, only_sentence)
    print("finish load train_phrases")
    valid_trees = sst_load_trees(valid_file_name)
    valid_phrases = sst_get_phrases(valid_trees, valid_ratio, is_binary, only_sentence or is_valid)
    if is_valid == False:
      train_phrases = train_phrases + valid_phrases
      valid_phrases = None
    test_trees = sst_load_trees(test_file_name)
    test_phrases = sst_get_phrases(test_trees, valid_ratio, is_binary, only_sentence=True)
    print("finish load test_phrases")

    cache["train"] = train_phrases
    cache["valid"] = valid_phrases
    cache["test"] = test_phrases
  else:
    train_phrases = cache["train"]
    valid_phrases = cache["valid"]
    test_phrases = cache["test"]
  #get word_id_dict
  word_id_dict["<pad>"] = 0
  word_id_dict["<unknown>"] = 1
  load_word_num_dict(train_phrases, word_num_dict)
  print("finish load train words: {0}".format(len(word_num_dict)))
  if valid_phrases != None:
    load_word_num_dict(valid_phrases, word_num_dict)
  load_word_num_dict(test_phrases, word_num_dict)
  print("finish load test words: {0}".format(len(word_num_dict)))
  word_id_dict = get_word_id_dict(word_num_dict, word_id_dict, min_count)
  print("after trim words: {0}".format(len(word_id_dict)))

  word_embed_model, unknown_word_embed = load_embedding(embedding_model)
  embedding = {}
  for model_name in word_embed_model:
    embedding[model_name] = np.random.random([len(word_id_dict), embed_dim]).astype(np.float32) / 2.0 - 0.25
    embedding[model_name] = init_trainable_embedding(embedding[model_name], word_id_dict,
                            word_embed_model[model_name], unknown_word_embed) 

  embedding["none"] = np.random.random([len(word_id_dict), embed_dim]).astype(np.float32) / 2.0 - 0.25
  embedding["none"][0] = np.zeros([embed_dim])
  embedding["field"] = np.random.random([slot_num + 1, embed_dim]).astype(np.float32) / 2.0 - 0.25
  embedding["field"][0] = np.zeros([embed_dim])
  embedding["pos"] = np.random.random([max_input_length, embed_dim]).astype(np.float32) / 2.0 - 0.25

  print("finish initialize word embedding")

  images["train"], images["train_bow_ids"], labels["train"], mask["train"] = sst_get_trainable_data(
          train_phrases, word_id_dict, word_embed_model, 1, max_input_length, is_binary)
  images["test"], images["test_bow_ids"], labels["test"], mask["test"] = sst_get_trainable_data(
          test_phrases, word_id_dict, word_embed_model, 2, max_input_length, is_binary)
  if valid_phrases != None:
    images["valid"], images["valid_bow_ids"], labels["valid"], mask["valid"] = sst_get_trainable_data(
            valid_phrases, word_id_dict, word_embed_model, 3, max_input_length, is_binary)
  else:
    images["valid"], images["valid_bow_ids"], labels["valid"], mask["valid"] = None, None, None, None

  return images, mask, labels, embedding
