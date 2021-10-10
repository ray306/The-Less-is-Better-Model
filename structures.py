from itertools import chain
from collections import Counter
import numpy as np

class TrieList:
    word_end = -1

    def __init__(self, lexicon=[]):
        """
        Initialize your data structure here.
        """
        self.root = {}
        self.word_end = -1
        self.par_list = []
        self.par_dict = dict()

        self.relationship = dict()
        self.root['skipgram'] = dict()

        for i in lexicon:
            if i not in self:
                self.append(i)

    def insert(self, index, word, note=True):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        curNode = self.root
        for c in word:
            if not c in curNode:
                curNode[c] = {}
            curNode = curNode[c]

        if self.word_end in curNode:
            raise Exception(f'already has "{word}"')
        else:
            curNode[self.word_end] = note
        
        if index == -1:
            self.par_list.append(word)
            self.par_dict[word] = len(self.par_list) - 1
        else:
            self.par_list.insert(index, word)
            par_dict[word] = index
       
    def append(self, word, note=True):
        self.insert(-1, word, note)
                
    def _del(self, word): # delete from the trie and dict
        # print(word)
        curNode = self.root
        def f(word, curNode):
            if len(word) == 0:
                del curNode[-1]
            else:  
                f(word[1:], curNode[word[0]])
                if len(curNode[word[0]]) == 0:
                    del curNode[word[0]]

        f(word, curNode)
        del self.par_dict[word]

    def _del(self, word): # delete from the trie and dict
        # print(word)
        curNode = self.root
        def f(word, curNode):
            if len(word) == 0:
                del curNode[-1]
            else:  
                f(word[1:], curNode[word[0]])
                if len(curNode[word[0]]) == 0:
                    del curNode[word[0]]

        f(word, curNode)
        del self.par_dict[word]
    
    def pop(self, index=-1):
        word = self.par_list.pop(index)
        self._del(word)
        return word

    def exchange(self, i, j):
        self.par_list[j], self.par_list[i] = self.par_list[i], self.par_list[j]

    def move(self, i, step):
        w = self.par_list[i]

        if step<0:
            if i!=0:
                self.par_list[i+step+1: i+1] = self.par_list[i+step: i] 
                self.par_list[i+step] = w
                self.par_dict[w] = i + step
            
        elif step>0:
            if i+step<len(self.par_list):
                self.par_list[i: i+step] = self.par_list[i+1: i+step+1]
                self.par_list[i+step] = w
                self.par_dict[w] = i + step
            else:
                self.par_list[i:] = self.par_list[i+1:]
                self._del(w)

    def group_move(self, words, update_rate, pre_update=True):
        if pre_update:
            self.update_ind() # make "index_with_prior" works

        offset = 0
        for w, e in words:
            # print(w)
            if w in self.par_dict:
                raw_ind = self.index_with_prior(w, offset)

                if raw_ind is not None and raw_ind > 0:
                    if e == 1: 
                        offset += 1

                    step = e * (int((raw_ind) * update_rate) + 1)                        
                    self.move(raw_ind, step)

        self.update_ind()

    def remove(self, word):
        ind = self.index_with_prior(word, nothing=-1)

        if ind != -1:
            self._del(word)
            self.par_list[ind: ind + 1] = []
    
    def group_remove(self, words):
        if len(words) > 0:
            words_ = []
            for w in words: 
                if w in self.par_dict:
                    self._del(w)
                    words_.append(w)
        
            # li = np.array(self.par_list,dtype=object)
            # self.par_list = li[~np.in1d(li, words_)].tolist()

            w_set = set(words_)
            self.par_list = [i for i in self.par_list if i not in w_set]

            self.update_ind()

    # def group_remove(self, words):
    #     if len(words) > 0:
    #         words_ = []
    #         for w in words: 
    #             if w in self.par_dict:
    #                 self.remove(w)
                    
    #         self.update_ind()

    def index(self, word):
        index = self.par_list.index(word)
        return index

    def index_with_prior(self, word, offset=0, nothing=False):
        try:
            prior_ind = self.par_dict[word]
            if offset == 0:
                return self.par_list.index(word, prior_ind)
            else:
                start = prior_ind - offset 
                if start < 0:
                    start = 0
                return self.par_list.index(word, start)
        except:
            if nothing==False:
                raise
            return nothing

    def update_ind(self):
        self.par_dict = dict(zip(self.par_list, range(len(self.par_list)))) # speed up 1.5x~2x than the next line
        # self.par_dict = dict((word,ind) for ind,word in enumerate(self.par_list))
        # self.relationship = dict((k,v) for k,v in self.relationship.items() if k in self.par_dict)

    def be_a_part(self, string):
        if not self.search(string):
            return False

        curNode = self.root
        for c in string:
            curNode = curNode[c]
        return (len(curNode) > 1)

    def startwith(self, string):
        if not self.search(string):
            return []
        
        chunks = []
        curNode = self.root
        for c in string:
            curNode = curNode[c]
            
        def f(pre, curNode):
            if len(curNode) == 1:
                chunks.append(pre)
            for i in curNode:
                if i!=-1:
                    f(pre+i, curNode[i])
        f(string, curNode)
        return chunks

    def has_a_larger(self, string):
        if not self.search(string):
            return None
 
        curNode = self.root
        for c in string:
            curNode = curNode[c]
            
        def f(pre, curNode):
            subs = [i for i in curNode if i!=-1]
            if len(subs) == 0:
                return pre
            elif len(subs) == 1:
                for i in subs:
                    return f(pre+i, curNode[i])
            else:
                return None

        if curNode == {-1: True}:
            return None
        else:
            return f(string, curNode)

    def search(self, word):
        return word in self.par_dict

    def skipgram_match(self, chunks):
        curNode = self.root['skipgram']
        gram = []

        for c in chunks:
            if c in curNode:
                gram.append(c)
                if len(gram) > 1:
                    break
                else:
                    curNode = curNode[c]
            elif len(gram) > 0:
                gram.append(c)
        else:
            gram = []

        if len(gram) > 2:
            return [gram[0],gram[-1]], gram[1:-1], chunks[chunks.index(gram[0])+1:]
        else:
            return None, None, None

    def match(self, string):
        curNode = self.root
        words = ['','']
        for c in string:
            if not c in curNode:
                if -1 in curNode:
                    return words[-1]
                else:
                    return words[-2]
                
            if -1 in curNode:
                words.append(words[-1]) # we know there was a chunk, now we make a new hypothesis
                
            words[-1] += c
            curNode = curNode[c]
        if -1 in curNode:
            return words[-1]
        else:
            return words[-2]

    def match_two(self, string):
        curNode = self.root
        words = ['','']
        for c in string:
            if not c in curNode:
                if -1 in curNode:
                    if len(words) > 1:
                        return words[-2:]
                    else:
                        return ['',words[-1]]
                else:
                    if len(words) > 2:
                        return words[-3:-1]
                    else:
                        return ['','']

            if -1 in curNode:
                words.append(words[-1]) # we know there was a chunk, now we make a new hypothesis

            words[-1] += c

            curNode = curNode[c]
        if -1 in curNode:
            return words[-2:]
        elif len(words) > 2:
            return words[-3:-1]
        else:
            return ['','']
        
    def match_2nd(self, string):
        curNode = self.root
        words = ['','']
        for c in string:
            if -1 in curNode:
                words.append(words[-1]) # we know there was a chunk, now we make a new hypothesis
            if not c in curNode:
                if -1 in curNode:
                    return words[-2]
                elif len(words) > 3:
                    return words[-3]
                else:
                    return ''
            words[-1] += c
            curNode = curNode[c]
        if -1 in curNode:
            return words[-2]
        elif len(words) > 3:
            return words[-3]
        else:
            return ''

    def __iter__(self):
        return iter(self.par_list)
            
    def __contains__(self, word):
        return self.search(word)
    
    def __repr__(self):
        return str(self.par_list[:100])+'...'
    
    def __len__(self):
        return len(self.par_list)
    
    def __set__(self):
        return set(self.par_list)
    
    def __getitem__(self, key):
        return self.par_list[key]
    
    def __setitem__(self, key, value):
        self._del(self.par_list[key])

        curNode = self.root
        for c in value:
            if not c in curNode:
                curNode[c] = {}
            curNode = curNode[c]
          
        curNode[self.word_end] = True

        self.par_list[key] = value

    def __copy__(self):
        return self.copy()

    def copy(self):
        new = TrieList(self.par_list)
        new.relationship = new.relationship
        return new

    def debug(self, mode):
        if mode == 'dict not root':
            return [i for i in self.par_dict if not self.match(i)==i]
        elif mode == 'dict not list':
            return [i for i in self.par_dict if i not in self.par_list]
        elif mode == 'list not root':
            return [i for i in self.par_list if not self.match(i)==i]
        elif mode == 'list not dict':
            return [i for i in self.par_list if i not in self.par_dict]
        else:            
            chunks = []
            curNode = self.root

            def f(pre, curNode):
                for i in curNode:
                    if i!=-1:
                        f(pre+i, curNode[i])
                    else:
                        chunks.append(pre)
            f('', curNode)
            if mode == 'root not list':
                li = [i if isinstance(i,str) else ''.join(i) for i in self.par_list]
            elif mode == 'root not dict':
                li = [i if isinstance(i,str) else ''.join(i) for i in self.par_dict]
            return [i for i in chunks if i not in li]

