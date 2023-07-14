class TrieList:
    def __init__(self, lexicon=[]):
        self.root = {}
        self.token_end = -1
        self.list = []
        self.dict = dict()

        for i in lexicon:
            if i not in self:
                self.append(i)

    # insert a token into the trie, dict, and list (to a given index)
    def insert(self, index, token, note=True):
        # Insert a token into the trie.
        node = self.root
        for c in token:
            if not c in node:
                node[c] = {}
            node = node[c]

        if self.token_end in node:
            raise Exception(f'already has "{token}"')
        else:
            node[self.token_end] = note
        
        if index == -1:
            self.list.append(token)
            self.dict[token] = len(self.list) - 1
        else:
            self.list.insert(index, token)
            self.dict[token] = index
       
    # insert a token into the trie, dict, and list (to its end)
    def append(self, token, note=True):
        self.insert(-1, token, note)

    # delete a token from the trie and dict. NOTE: the list is not updated.
    def _del(self, token):
        node = self.root
        def f(token, node):
            if len(token) == 0:
                del node[-1]
            else:  
                f(token[1:], node[token[0]])
                if len(node[token[0]]) == 0:
                    del node[token[0]]

        f(token, node)
        del self.dict[token]

    # move a token in the list. `step` defines the move span.
    def move(self, i, step):
        w = self.list[i]

        if step < 0 and i != 0: # if the token is the first one, it cannot be moved begining-ward.
            if i+step < 0: # if the begining-ward step is too large, move the token to the head of the list
                self.list[1: i+1] = self.list[0: i]
                self.list[0] = w
                self.dict[w] = 0
            else:
                self.list[i+step+1: i+1] = self.list[i+step: i] # move the tokens between the original and the new position
                self.list[i+step] = w # move the token to the new position
                self.dict[w] = i + step # update the index of the token in the dict
        elif step > 0:
            if i+step < len(self.list):
                self.list[i: i+step] = self.list[i+1: i+step+1]
                self.list[i+step] = w
                self.dict[w] = i + step
            else: # if the tail-ward step is larger than the length of the list, remove the token from the list
                self.list[i:] = self.list[i+1:]
                self._del(w)

    # move a group of tokens in the list. `update_rate` defines the move span which is proportional to the index of the token.
    def batch_move(self, tokens, update_rate, pre_update=True):
        if pre_update: # if `dict` stores the correct index of each item in `list`, then we can skip this step.
            self.update_ind() # The more accurate the information in `dict`, the faster the query of `self.index()`.

        offset = 0
        for w, e in tokens:
            if w in self.dict:
                raw_ind = self.index(w, offset)
                if raw_ind is not None and raw_ind > 0:
                    # A tail-ward move of one token may reduce the index of other tokens by 1.
                    # So tail-ward moves are recorded in `offset`.
                    # A begining-ward move may also affect the indexes of other tokens, but does not confuse the search from a given position to the tail.
                    if e == 1: 
                        offset += 1

                    step = e * (int((raw_ind) * update_rate) + 1) # the movement span is proportional to the index of the token.              
                    self.move(raw_ind, step)

        self.update_ind()

    # remove a token from the trie, dict, and list.
    def remove(self, token):
        ind = self.index(token, nothing=-1)

        if ind != -1:
            self._del(token)
            self.list[ind: ind + 1] = []
    
    # remove a group of tokens from the trie, dict, and list.
    def batch_remove(self, tokens):
        if len(tokens) > 0:
            tokens_ = []
            # remove the tokens from the dict
            for w in tokens: 
                if w in self.dict:
                    self._del(w)
                    tokens_.append(w)
            # remove the tokens from the list
            w_set = set(tokens_)
            self.list = [i for i in self.list if i not in w_set]

            self.update_ind()

    # update the item indexes of the list in the dict
    def update_ind(self):
        self.dict = dict(zip(self.list, range(len(self.list)))) # speed up 1.5x~2x than "dict((token,ind) for ind,token in enumerate(self.list))"

    # return the index of the token in the list
    def index(self, token, offset=0, nothing=False):
        start = self.dict[token]  # get the stored index from the dict
        if offset != 0:
            start -= offset 
            if start < 0:
                start = 0

        try:
            return self.list.index(token, start)
        except: # if the token is not in the list
            if nothing==False:
                raise
            return nothing # return the content of `nothing` instead of raising an error
  
    # Returns the longest token in the string that matches the trie
    def match(self, string, last_n=1):
        node = self.root
        candidate = '' # the substring that matches the trie
        tokens = [''] # the matched tokens. '' is the default result.

        # go through the string, one character at a time
        for c in string:
            if not c in node: # if the trie doesn't have this character, no larger token will match
                break

            candidate += c # update the candidate
            node = node[c] # go to the next node
            if -1 in node: # if it is the end of a token, we confirm there was a matched token
                tokens.append(candidate)

        if last_n == 1: # return the last token
            return tokens[-1]
        else: # return the last tokens
            if len(tokens) < last_n: # if there are not enough tokens to return, pad it with empty strings
                tokens = [''] * (last_n - len(tokens)) + tokens
            return tokens[-last_n:]
            
    # if nothing matched, return the first symbol of the string
    def force_match(self, string):
        res = self.match(string)
        if res == '' and len(string) > 0:
            return string[0]
        else:
            return res
        
    # find the skipgrams that match the chunk list
    def skip_match(self, chunks):
        candidate = []
        for i in range(len(chunks)):
            node = self.root
            start = chunks[i]
            # if the current chunk matches a skipgram beginning, check if the rest of the chunks match the skipgram end
            if start in node:
                for end in node[start]:
                    if end in chunks[i+1:]:
                        candidate.append(((start, end), chunks[i+1: chunks.index(end, i+1)]))
        return candidate

    def copy(self):
        copied = TrieList(self.list)
        return copied
    
    def __iter__(self):
        return iter(self.list)
            
    def __contains__(self, token):
        return token in self.dict
    
    def __repr__(self):
        return str(self.list[:100])+'...'
    
    def __len__(self):
        return len(self.list)
    
    def __set__(self):
        return set(self.list)
    
    def __getitem__(self, idx):
        return self.list[idx]

    # # set a token on the given index. 
    # def __setitem__(self, idx, token):
    #     # _del the original token from the trie and dict
    #     self._del(self.list[idx])

    #     # set the token in the list
    #     self.list[idx] = token

    #     # add a new token to the trie and dict
    #     node = self.root
    #     for c in token:
    #         if not c in node:
    #             node[c] = {}
    #         node = node[c]
    #     node[self.token_end] = True

    #     self.dict[token] = idx
    

    # def be_a_part(self, string):
    #     if string not in self:
    #         return False

    #     node = self.root
    #     for c in string:
    #         node = node[c]
    #     return (len(node) > 1)

    # def has_a_larger(self, string):
    #     if string not in self:
    #         return None
 
    #     node = self.root
    #     for c in string:
    #         node = node[c]
            
    #     def f(pre, node):
    #         subs = [i for i in node if i!=-1]
    #         if len(subs) == 0:
    #             return pre
    #         elif len(subs) == 1:
    #             for i in subs:
    #                 return f(pre+i, node[i])
    #         else:
    #             return None

    #     if node == {-1: True}:
    #         return None
    #     else:
    #         return f(string, node)

    # # Returns the longest token in the string that matches the trie
    # def match(self, string, return_two=False):
    #     node = self.root
    #     tokens = ['',''] # the last element may not be the lasted matched token, but we will update it as a hypothesis

    #     # go through the string, one character at a time
    #     for c in string:
    #         if not c in node: # if the trie doesn't have this character, no larger token will match
    #             break

    #         if -1 in node: # if it is the end of a token, we confirm there was a matched token
    #             tokens.append(tokens[-1]) # now we make a new hypothesis 
    #         tokens[-1] += c # update the hypothesis
    #         node = node[c] # go to the next node

    #     if -1 not in node: # if the last character is not the end of a token, the last item in `tokens` is not a matched token
    #         tokens.pop()

    #     if return_two: # return the last two tokens
    #         return tokens[-2:] if len(tokens) > 1 else ['','']
    #     else: # return the last token
    #         return tokens[-1]

    #     # if return_two: # return the last two tokens
    #     #     if -1 in node: # if the last character is the end of a token, the last item in `tokens` is a candidate
    #     #         return tokens[-2:]
    #     #     elif len(tokens) > 2: # if the last character is not the end of a token, the second last item in `tokens` is a candidate
    #     #         return tokens[-3:-1]
    #     #     else:
    #     #         return ['','']
    #     # else: # return the last token
    #     #     if -1 in node:
    #     #         return tokens[-1]
    #     #     else:
    #     #         return tokens[-2]

    # # find the common bugs
    # def debug(self, mode):
    #     if mode == 'dict not root':
    #         return [i for i in self.dict if not self.match(i)==i]
    #     elif mode == 'dict not list':
    #         return [i for i in self.dict if i not in self.list]
    #     elif mode == 'list not root':
    #         return [i for i in self.list if not self.match(i)==i]
    #     elif mode == 'list not dict':
    #         return [i for i in self.list if i not in self.dict]
    #     else:            
    #         chunks = []
    #         node = self.root

    #         def f(pre, node):
    #             for i in node:
    #                 if i!=-1:
    #                     f(pre+i, node[i])
    #                 else:
    #                     chunks.append(pre)
    #         f('', node)
    #         if mode == 'root not list':
    #             li = [i if isinstance(i,str) else ''.join(i) for i in self.list]
    #         elif mode == 'root not dict':
    #             li = [i if isinstance(i,str) else ''.join(i) for i in self.dict]
    #         return [i for i in chunks if i not in li]