""" I would just ignore this whole class, it's just a lot of rules that work nicely for UMLS
once the software is trained the main thing are the context vectors.
"""

import numpy as np
import operator

class CatAnn(object):
    def __init__(self, umls, spacy_cat):
        self.umls = umls
        self._cat = spacy_cat


    def add_ann(self, name, tkns, doc, to_disamb, doc_words):
        one_tkn_upper = False
        name_case = True

        if len(tkns) == 1 and tkns[0].is_upper:
            one_tkn_upper = True
        for tkn in tkns:
            if not tkn.is_upper:
                name_case = False

        # Don't allow concatenation of tokens if len(name) < 5
        if not(len(name) < 6 and len(tkns) > 1):
            # Name must have > 3, if not disambiguation is a must
            if len(name) > 3:
                if len(self.umls.name2cui[name]) == 1:
                    cui = list(self.umls.name2cui[name])[0]

                    if len(name) < 6:
                        # Disambiguation needed if length of string < 6
                        # Case must agree or first can be lower_case
                        if not name_case or self.umls.name_isupper[name] == name_case:
                            if not name_case or (len(name) > 4):
                                # Means name is not upper, disambiguation is needed
                                n_words, words_cnt = self._n_words_appearing(name, doc, doc_words)
                                d = self.umls.cui2words[cui]
                                perc =  0
                                cnt = 0
                                if name in d:
                                    perc = d[name] / sum(d.values())
                                    cnt = d[name]
                                if (n_words > len(tkns) and words_cnt > 5) or (perc > 0.2 or cnt > 5):
                                    self._cat._add_ann(cui, doc, tkns, acc=1, name=name)
                                else:
                                    to_disamb.append((list(tkns), name))
                            else:
                                # Was lowercase and shorther than 5 characters
                                to_disamb.append((list(tkns), name))
                        else:
                            # Case dosn't match add to to_disamb
                            to_disamb.append((list(tkns), name))
                    else:
                        # Longer than 5 letters, just add concept
                        cui = list(self.umls.name2cui[name])[0]
                        self._cat._add_ann(cui, doc, tkns, acc=1, name=name)
                else:
                    # Means we have more than one cui for this name
                    scores = self._scores_words(name, doc, doc_words, tkns)
                    acc = self.softmax(scores.values())
                    if len(name) < 6:
                        if self.umls.name_isupper[name] == name_case or (not name_case and len(name) > 3):
                            # Means match is upper in both cases, tag if acc > 0.5
                            if acc > 0.5:
                                cui = max(scores.items(), key=operator.itemgetter(1))[0]
                                self._cat._add_ann(cui, doc, tkns, acc=acc, name=name)
                            else:
                                to_disamb.append((list(tkns), name))
                        else:
                            to_disamb.append((list(tkns), name))
                    else:
                        # We can be almost sure that everything is fine, threshold of 0.1
                        if acc > 0.1:
                            cui = max(scores.items(), key=operator.itemgetter(1))[0]
                            self._cat._add_ann(cui, doc, tkns, acc=acc, name=name)
                        else:
                            to_disamb.append((list(tkns), name))
            else:
                to_disamb.append((list(tkns), name))


    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        x = list(x)
        if not any(x):
            return 0
        al = True
        for i in x:
            if i > 0:
                al = False
        if al:
            return 0
        e_x = np.exp(x / np.max(x))
        return max(e_x / e_x.sum())


    def _scores_words(self, name, doc, doc_words, tkns):
        scores = {}

        name_cnt = self.umls.name2cnt[name]
        sm = sum(name_cnt.values())

        for cui in self.umls.name2cui[name]:
            score = 0
            n = 0
            flag = False

            if len(tkns) == 1 and len(name) > 3:
                # Only prefered names are taken into account
                if cui in self.umls.cui2pref_name:
                    if name != self.umls.cui2pref_name[cui]:
                        flag = True

            if not flag:
                for word in self.umls.cui2words[cui].keys():
                    if word in doc_words:
                        n += 1
                        score += self.umls.cui2words[cui][word] / self.umls.vocab[word]
                if n > 0:
                    score = score / n

                # Add proportion for name count
                score = (score + (name_cnt[cui] / sm)) / 2

                # Check is this the prefered name for this concept
                if len(name) > 3:
                    if cui in self.umls.cui2pref_name:
                        if name == self.umls.cui2pref_name[cui]:
                            score = score + 1

                """ This improves the acc by around 1% but reduces the speed by 50%
                if len(name) < 6:
                    #cntx_score = self._cat._calc_acc(cui, doc, tkns)
                    #if cntx_score != -1:
                    #    score = (score + cntx_score) / 2
                    pass
                """
            scores[cui] = score
        return scores


    def _n_words_appearing(self, name, doc, doc_words):
        cui = list(self.umls.name2cui[name])[0]

        n = 0
        cnt = 0
        for word in self.umls.cui2words[cui].keys():
            if word in doc_words:
                n += 1
                cnt += doc_words.count(word)
        return n, cnt