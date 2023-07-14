
from typing import Dict, List, Union
from dataclasses import dataclass, field

import random
import time
import math
from statistics import mean

import structures
import preprocessing

import importlib
importlib.reload(preprocessing)
importlib.reload(structures)

from preprocessing import create_lexicon, merge_token_list, Corpus

@dataclass
class Model():
    corpus: Corpus = field(repr=False) # the corpus containing the training and testing data
    lexicon: Union[structures.TrieList, list, None] = field(default=None, repr=False) # the lexicon

    in_rate: float = 0.25 # ["α" in paper] between 0 and 1
    # The combination of two chunks have a chance to be the new chunk for memorizing .
    # 'in_rate' determines the sampling probability for memorizing the combined chunks.
    # More frequent combinations are more likely to be memorized.
    # A larger value means the more infrequent combinations will be memoried.
    # A value of 0.25 indicates a 25% chance of selecting the combined chunk for memorization.
    
    out_rate: float = 0.0001 # ["ω" in paper] between 0 and 1
    # In each step, The chunks at the lexicon tail will be sent to probation. 
    # 'out_rate' determines the percentage of chunks selected from the lexicon tail.
    # They will not be forgotten immediately but will be sent to probation.
    # A value of 0.01 indicates that 1% of the chunks from the lexicon tail will be selected for further evaluation. 
    
    life: int = 10 # ["τ0" in paper] >= 1
    # For the new memorized chunk or the chunks at the lexicon tail, they will be sent to probation for further evaluation.
    # `life` determines the probation period / the lifespan of a chunk (in steps)
    # A value of 10 indicates that a chunk will be on probation for 10 steps.
    # Training on a larger corpus may need a larger `life` since a unit is likely to distribute more sparsely in a large corpus.

    update_rate: float = 0.1 # ["∆" in paper] between 0 and 1
    # The chunk's rank within the lexicon will be updated for rewarding or punishing it.
    # 'update_rate' determines the distance that a chunk will be moved (re-ranked) across during the update process (rewarding or punishing).
    # Deceasing the update rate will slow down the training speed, but also make the ranks of the chunks consistent with their frequencies and their ability to reduce the tokens.
    # A value of 0.1 indicates that the target chunk will be shifted across 10% of its current rank in the lexicon.
    
    # P.S. `in_rate` and `update_rate` mainly affect the training speed, while `out_rate` and `life` mainly affect lexicon size

    max_chunk_len: Union[None, int] = None # the maximum length of a chunk to be memorized  (>= 1)
    # A value of None indicates that there is no limit on the chunk length.

    inspect_context: bool = True # whether to inspect the context of the chunk during memorizing
    # If `inspect_context` is True, the chunk will pass the probation period only if the context varies; otherwise, the chunk will be memorized regardless of the context.
    # For example, if the chunk is "the", it will pass the probation period only if the context is not "the" (e.g., "the cat" or "the dog").
    # As an opposite example, if the chunk is "the ca", but the model can only find "t" as its context, "the ca" will NOT pass the probation period
    
    # use_skipgram: bool = True # If `use_skipgram` is True, the model will use the skipgram method to find the new tokens during memorizing.
    
    random_seed: int = 0 # the memorizing has some random factors, fix the seed to control the result
    probation_pool: Dict = field(default_factory=dict, repr=False) # store the chunks on probation
    reward_pool: List = field(default_factory=list, repr=False) # store the rewards/punishments for the chunks
    
    logs: Dict = field(default_factory=lambda:
                        {'eval_index': [], 'time_cost': [], 'rewards': {}, 'note': []}, 
                       repr=False) # store the logs during learning

    def __post_init__(self):
        if self.random_seed != None:
            random.seed(self.random_seed)
        self.corpus.reset_iterate() # reset the corpus iterator to the beginning

        # init the Lexicon from the input
        if self.lexicon is None:
            self.lexicon = create_lexicon()
        elif isinstance(self.lexicon, list):
            self.lexicon = create_lexicon(self.lexicon)
        else:
            self.lexicon = self.lexicon.copy() # make a copy so the change of `self.lexicon` during learning won't influence the input `lexicon`

    'Read the corpus (segment) and update Lexicon'
    def reading(self, article):
        for sent in article:
            chunks_in_sent = []
            to_eval = dict()
            loc = 0

            while loc < len(sent):
                # evalute the chunks stored in `to_eval` by examining the 
                self.evaluate(to_eval, sent, loc)

                # The `last_n` tell the method to return the largest and second largest sequences, starting from the beginning of the string, that matches any stored chunks in the lexicon.
                # So there will be two options for the first chunk in the current segmentation: the default chunk `chunk` and the alter chunk `chunk_smaller`
                chunk_smaller, chunk = self.lexicon.match(sent[loc:], last_n=2)

                # update the default segmentation in `to_eval`
                for c in list(to_eval.keys()):
                    if len(chunk) > 0:
                        to_eval[c][1].append(chunk)
                    else:
                        to_eval[c][1].append(sent[loc])

                chunk_last = chunks_in_sent[-1] if len(chunks_in_sent) > 0 else ''
                chunk_before_last = chunks_in_sent[-2] if len(chunks_in_sent) > 1 else ''

                # The presence of the current chunk in probation list means that it occurs twice with in its lifespan, 
                # then it has a chance to be released from probation.
                if chunk in self.probation_pool:
                    if self.inspect_context and self.probation_pool[chunk][1] != None:
                        # `chunk_next` is the largest sequence, starting from the end of the current chunk, that matches any stored chunks in the lexicon.
                        # if nothing matches, `chunk_next` will be the first symbol of the string (implemented by `force_match`) in order to offer the correct context.
                        chunk_next = self.lexicon.force_match(sent[loc+len(chunk):])
                        old_last, old_next = self.probation_pool[chunk][1]

                        # if the context vary (or the context is the sentence boundary), release the target chunk from probation; 
                        # otherwise, memorize the merge of chunk and its fix context (and start the new probation)
                        if old_last == old_next == chunk_last == chunk_next == '':
                            pass
                        else:
                            extended = False
                            if (chunk_last.endswith(old_last) and old_last != '') or \
                               (old_last.endswith(chunk_last) and chunk_last != ''): # the left context is extended
                                self.memorize(chunk_last + chunk, (chunk_before_last, chunk_next))
                                extended = True
                            if (chunk_next.startswith(old_next) and old_next != '') or \
                               (old_next.startswith(chunk_next) and chunk_next != ''): # the right context is extended
                                chunk_after_next = self.lexicon.force_match(sent[loc + len(chunk) + len(chunk_next):]) # 
                                self.memorize(chunk + chunk_next, (chunk_last, chunk_after_next))
                                extended = True
                            
                            if not extended:
                                # print(f'|{old_last}|{chunk}|{old_next}|%%|{chunk_last}|{chunk}|{chunk_next}|')
                                self.reward_pool.append((chunk, -1))
                                del self.probation_pool[chunk] # release the chunk from probation

                                # if self.use_skipgram:
                                #     context_ = ('**'+chunk_last, '**'+chunk_next)
                                #     self.memorize(context_, None)
                                
                                #     context_ = ('**'+old_last, '**'+old_next)
                                #     if context_ in self.lexicon:
                                #         self.reward_pool.append((context_, -1))
                                #         if context_ in self.probation_pool:
                                #             del self.probation_pool[context_]
                                #     else:
                                #         self.memorize(context_, None)

                                
                    else:
                        del self.probation_pool[chunk] # release the chunk from probation
 
                # memorizing
                if len(chunk) > 0:
                    if len(chunk) > 1 and len(chunk_smaller) > 0: # start a new evaluation; otherwise, the evaluation is a waste of time
                        to_eval[chunk] = [loc + len(chunk_smaller), [chunk], [chunk_smaller]]
                    # memorize the combination of the last chunk and the current chunk (if they are lucky)
                    if random.random() < self.in_rate:
                        chunk_next = self.lexicon.force_match(sent[loc+len(chunk):])
                        self.memorize(chunk_last + chunk, (chunk_before_last, chunk_next))
                else:
                    # memorize the symbol on the current location
                    chunk = sent[loc]
                    chunk_next = self.lexicon.force_match(sent[loc+1:])
                    self.memorize(chunk, (chunk_last, chunk_next))

                chunks_in_sent.append(chunk)
                loc += len(chunk)

        # if self.use_skipgram:
        #     candidates = self.lexicon.skip_match(['**'+c for c in ['']+chunks_in_sent+['']])
        #     for (context_0, context_1), fill in candidates:
        #         if (context_0, context_1) != ('**', '**'):
        #             fill = ''.join(map(lambda x:x[2:], fill)) # remove the '**' from each token
        #             context = (context_0[2:], context_1[2:])
        #             self.memorize(fill, context)
                
        # update the chunks in Lexicon by processing them in batch'
        forget_list = self.batch_forget()
        self.batch_rerank(forget_list)
        self.batch_probation_mark()

    'Store the chunk in Lexicon'
    def memorize(self, chunk, context):
        if self.max_chunk_len and len(chunk) > self.max_chunk_len: # if `max_chunk_len` is not None, limit the chunk length
            return

        if chunk and (chunk not in self.lexicon):
            self.lexicon.append(chunk)
            self.probation_pool[chunk] = (self.life, context) # the first item: init the lifespan of the target chunk; the second item: record the previous chunk and the subsequent chunk of the target. 
            
    'Mark the chunks on the tail of Lexicon as bad chunks and place them on probation'
    def batch_probation_mark(self):
        for ind in range(int((1-self.out_rate)*len(self.lexicon)), len(self.lexicon)):
            _chunk_ = self.lexicon[ind]
            if _chunk_ not in self.probation_pool:
                self.probation_pool[_chunk_] = (self.life, None)

    'remove the junk chunks'
    def batch_forget(self):
        forget_list = []

        for _chunk_ in list(self.probation_pool.keys()):
            current_life, context = self.probation_pool[_chunk_]
            current_life -= 1
            if current_life < 1: # if the chunk's lifespan has reached 0, it will be removed
                del self.probation_pool[_chunk_]
                forget_list.append(_chunk_)
                
                # if self.use_skipgram and context != None:
                #     context_ = ('**'+context[0], '**'+context[1])
                #     if context_ in self.lexicon:
                #         self.reward_pool.append((context_, 1))

            else: # if not, reduce its lifespan
                self.probation_pool[_chunk_] = current_life, context

        self.lexicon.batch_remove(forget_list)

        return forget_list

    'Reorder chunks by if their are rewarded or punished'
    def batch_rerank(self, forget_list):
        forget_set = set(forget_list)
        self.reward_pool = [(w,e) for w,e in self.reward_pool if w not in forget_set]
        self.lexicon.batch_move(self.reward_pool, self.update_rate, pre_update=False)
        self.reward_pool = []

    'Decompose the chunks to subchunks if there are subchunk candidates'
    def find_subs(self, large_chunk, level=2):
        # search all possible 2-fold segmentations
        chunk_1 = large_chunk
        subs = [] # store the subchunks and their ranks
        while 1:
            chunk_1 = self.lexicon.match(chunk_1[:-1]) # the first part
            chunk_2 = large_chunk[len(chunk_1):] # the second part
            
            if len(chunk_1) > 0 and chunk_2 in self.lexicon:
                chunk_1_index, chunk_2_index = self.lexicon.index(chunk_1), self.lexicon.index(chunk_2)
                subs.append((chunk_1, chunk_2, (chunk_1_index, chunk_2_index)))
            
            if len(chunk_1) <= 1: # the search is done
                break
        
        # select the segmentation in which the ranks of all subchunks are lower than the rank of the original chunk;
        # if no one selected, return the original chunk
        if len(subs) > 0:
            sub = sorted(subs, key=lambda x:x[2])[0]
            if max(sub[2]) < self.lexicon.index(large_chunk, nothing=len(self.lexicon)):
                if level == 1:
                    return sub[:2]
                else: # continue the decompostion 
                    return self.find_subs(sub[0], level-1) + self.find_subs(sub[1], level-1)
            else:
                return (large_chunk,)
        else:
            return (large_chunk,)

    'detect the next chunk from the given location and update the input'
    def next_segment(self, sent, loc, chunks):
        chunk = self.lexicon.match(sent[loc:])
        if len(chunk) == 0:
            chunk = sent[loc]

        loc += len(chunk)
        chunks.append(chunk)
        return loc, chunks

    'check if the the default segmentation is redundant'
    def alter_is_better(self, chunks_in_default, chunks_in_alter):
        # trick or treat? let's start!
        N_1 = len(chunks_in_default)
        N_2 = len(chunks_in_alter)

        # the default segmentation is redundant if it is longer -- or has larger ranks -- than the alter segmentation
        alter_is_better = (N_1 > N_2) or \
                           (N_1 == N_2 and \
                                sum(self.lexicon.index(i) 
                                        for i in chunks_in_default if i in self.lexicon) > \
                                sum(self.lexicon.index(i) 
                                        for i in chunks_in_alter if i in self.lexicon))
        return alter_is_better

    'evaluate the given chunk as good/bad by whether it reduces the number of tokens'
    def evaluate(self, to_eval, sent, default_end):
        for chunk, (alter_end, chunks_in_default, chunks_in_alter) in list(to_eval.items()):
            # We already have a default segmentation by `reading()`, and we are trying to find an alter segmentation.
            # start from the end of the alter segmentation, add the subsequent chunks to `chunks_in_alter`
            while alter_end < default_end:
                alter_end, chunks_in_alter = self.next_segment(sent, alter_end, chunks_in_alter)
                    
            if alter_end == default_end: # the default/alter segmentations reach the same end
                # reward/punish the chunks
                if self.alter_is_better(chunks_in_default, chunks_in_alter):
                    self.reward_pool.append((chunks_in_alter[0], -1)) # -1 is reward: the rank value will be reduced
                    self.reward_pool.append((chunk, 1)) # 1 is punish: the rank value will be added
                else:
                    self.reward_pool.append((chunk, -1))

                del to_eval[chunk]
            else: # the alter segmentation has not reached default segmentation; cannot start the evaluation
                to_eval[chunk] = alter_end, chunks_in_default, chunks_in_alter

    'learn the generated corpus sample'
    def learn(self, step_id, article_length_max=500, report_interval=100, sequential_read=True, loop=True):
        '''
        Extract some sentences (default: 500) from the corpus as an article, and train on the article.
        Show evaluation on the test corpus every `report_interval` steps.

        Parameters
        ----------
        step_id: int
            The current step id.
        article_length_max: int, None (default: 500)
            The max number of sentences to be extracted from the corpus as an article.
            If the corpus is exhausted, the article will be shorter than this value.
            If None, extract an whole article (if there is the level of article in the corpus) or a single sentence as the article(if not).
        report_interval: int (default: 100)
            The interval of steps to show the evaluation on the test corpus.
        sequential_read: bool (default: True)
            If True, extract article from the corpus sequentially; if False, randomly.
        loop: bool (default: True)
            If True, the corpus will be read repeatedly; if False, an Exception will be raised when the corpus is exhausted.

        Returns
        -------
        None
        '''

        # # In the `sequential_read` mode, the parameter `life` should be set to a value less than the total number of steps required to complete a full iteration through the entire training corpus. 
        # # The newly memorized chunks are the combination of the pre-existing chunks, but these could end up being accidental combinations rather than meaningful combinations. 
        # # To remove the accidental combinations, the new chunks are sent to probation to see whether they can occur again. 
        # # However, if the lifespan is too long, it guarantees the recurrence of the probationary chunks (by reading the same article twice), 
        # #                                       so NO chunks will be forgotten during the learning.
        # life_allowed = math.ceil(len(self.corpus.train)/article_length_max) # the steps taken to complete a full iteration of the training corpus.
        # if self.life > life_allowed:
        #     print('Warning: the parameter `life` or `article_length_max` is too large. The chunks can\'t be forgotten during the training.\n')

        start_time = time.time()
        
        # the pipeline for learning from queue import Queue
        article_raw = self.corpus.extract_article(article_length_max, sequential_read, loop) # extract some sentences from the corpus as an article
        article = merge_token_list(article_raw) # merge the token lists as strings for each sentence
        self.reading(article) # read and learn the article 

        self.logs['time_cost'].append(time.time()-start_time)

        # The newly memorized chunks are the combination of the pre-existing chunks, but these could end up being accidental combinations rather than meaningful combinations. 
        # To remove the accidental combinations, the new chunks are sent to probation to see whether they can occur again. 
        # However, if the lifespan or article_length_max is too long, it increases the probability of reading the same article twice in a short period of time, 
        #                                       so NO chunks learned from that article will be forgotten during the learning.
        # self.recently_read_articles = self.recently_read_articles[-(self.life-1):] # keep the last (`life`-1) articles
        # pos_a, pos_s = self.corpus.current_article_position, self.corpus.current_sentence_position
        # for pos_a_, pos_s_ in self.recently_read_articles:
        #     if pos_a == pos_a_ and pos_s_ <= pos_s < pos_s_+article_length_max: # the article has been read recently
        #         print('Warning: the current article had been read recenlty. Some bad chunks can\'t be forgotten.\nDecrease the parameter `life` or `article_length_max`.')
        # self.recently_read_articles.append((pos_a, pos_s))

        if report_interval != None and step_id % report_interval == 0:
            self.show_evaluation(step_id)

    'remove the chunks in probation from the lexicon'
    def clean_lexicon(self):
        self.lexicon.batch_remove(list(self.probation_pool.keys()))

    'apply the Lexicon to the given sentence samples (single sentence must be contained in a list)'
    def apply(self, article=[], seg_deeper=False, flatten=False):
        '''
        Segment the test corpus or the given article (single sentence must be contained in a list).

        Parameters
        ----------
        article: list (default: [])
            The sentence samples to be tested.
            if empty, the test corpus will be used.
        seg_deeper: bool (default: False)
            If True, the segmentation will be decomposed into deeper levels (aka. subchunks).
        flatten: bool (default: True)
            If True, the level of sentence will be flattened and the result will be a list of tokens;
            if False, the result will be a list of sentences.
        
        Returns
        -------
        list
        '''
        if len(article) == 0:
            article = self.corpus.test

        # merge the token lists as strings for each sentence
        if isinstance(article[0], list):
            article = merge_token_list(article)

        segmented_article = []
        
        for sent in article:
            sent_chunks = []
            i = 0

            while i < len(sent):
                # The `last_n` tell the method to return the largest and second largest sequences, starting from the beginning of the string, that matches any stored chunks in the lexicon.
                # So there will be two options for the first chunk in the current segmentation: the default chunk `chunk` and the alter chunk `chunk_smaller`
                chunk_smaller, chunk = self.lexicon.match(sent[i:], last_n=2)

                if len(chunk) > 1 and len(chunk_smaller) > 0:
                    # continue the segmentation by considering these two options, each leading to a distinct path;
                    # stop the segmentation until the endpoints for both paths align.
                    chunks_in_default, chunks_in_alter = [chunk], [chunk_smaller]
                    default_end, alter_end = i + len(chunk), i + len(chunk_smaller)

                    while default_end != alter_end:
                        if default_end > alter_end:
                            alter_end, chunks_in_alter = self.next_segment(sent, alter_end, chunks_in_alter)
                        elif default_end < alter_end:
                            default_end, chunks_in_default = self.next_segment(sent, default_end, chunks_in_default)

                    # find the better segmentation path and store it in `sent_chunks`
                    if self.alter_is_better(chunks_in_default, chunks_in_alter):
                        sent_chunks += chunks_in_alter
                        i += sum(len(c) for c in chunks_in_alter)
                    else:
                        sent_chunks += chunks_in_default
                        i += sum(len(c) for c in chunks_in_default)

                    # if self.alter_is_better(chunks_in_default, chunks_in_alter):
                    #     sent_chunks.append(chunks_in_alter[0])
                    #     i += len(chunks_in_alter[0])
                    # else:
                    #     sent_chunks.append(chunks_in_default[0])
                    #     i += len(chunks_in_default[0])
                else:
                    sent_chunks.append(sent[i]) 
                    i += 1    
        
            segmented_article.append(sent_chunks)
            
        # Decompose the chunks to their subchunks
        if seg_deeper:
            segmented_article = [[sub_c for c in sent for sub_c in self.find_subs(c)] for sent in segmented_article]
            
        # Flatten the sentences level
        # which means the structure [sentence1->[tokens], sentence2->[tokens]]  will be the structure [tokens]  
        if flatten:
            segmented_article = [c for sent in segmented_article for c in sent]
            
        return segmented_article

    'display the segmentation on the test corpus or given article'
    def segment_and_show(self, article=[], show_orignal=True, seg_deeper=False):
        '''
        Display the segmentation on the test corpus or given article.

        Parameters
        ----------
        article: list (default: [])
            The sentence samples to be tested.
            if empty, the test corpus will be used.
        show_orignal: bool (default: True)
            If True, the orignal segmentations will be shown above the LiB segmentations.
        seg_deeper: bool (default: False)
            If True, the segmentation will be decomposed into deeper levels (aka. subchunks).
        
        Returns
        -------
        None
        '''
        if len(article) == 0:
            article = self.corpus.test

        # for rich display
        ENDC = '\033[0m'
        BOLD = '\033[1m'

        # iterate on each sentence
        for sent in article:
            if show_orignal:
                if isinstance(sent, list):
                    print('|'.join(sent))
                else:
                    print('Warning: You can\'t see orignal segmentation because the corpus sentences (the parameter `article`) are strings. Use `show_orignal=False` to show the LiB segmentation only.')
            print(BOLD + '|'.join(self.apply([sent], seg_deeper=seg_deeper, flatten=True)) + ENDC)
            print()

    'show the evaluation metrics on the test corpus'
    def show_evaluation(self, step_id):
        '''
        Display the evaluation metrics on the test corpus.

        Parameters
        ----------
        step_id: int
            The current step id.
        
        Returns
        -------
        None
        '''
        article = merge_token_list(self.corpus.test)
        chunks = self.apply(article, flatten=True)
        
        # codebook = ''.join(list(set(chunks)))
        # chunk_counter = Counter(chunks)
        # char_counter = Counter(codebook)
        # LM = -sum(char_counter[t]*math.log2(char_counter[t]/len(codebook)) for t in char_counter)
        # LD = -sum(chunk_counter[t]*math.log2(chunk_counter[t]/len(chunks)) for t in chunk_counter)

        avg_chunk_len = (sum(len(w) for w in chunks)/len(chunks))
        eval_index = avg_chunk_len/math.log2(len(self.lexicon))  # better result: longer chunk length/fewer chunk tokens in corpus; fewer chunk tokens in Lexicon 
        mem_avg = mean([len(i) for i in self.lexicon.list])
        self.logs['eval_index'].append(eval_index)

        print(f'{step_id}\t  LexiconSize: {len(self.lexicon)}  TypeLength: {mem_avg:.1f}  TokenLength: {avg_chunk_len:.1f}\t  EvalIndex: {eval_index:.3f}')
        if step_id==0:
            print('\n')
        else:
            print(f'\t\t\t\t\t\t\t\t\tΔ: {(self.logs["eval_index"][-1]-self.logs["eval_index"][-2])/self.logs["eval_index"][-2]*100:.1f}%')

    'Save the lexicon'
    def save_lexicon(self, path, plain_text=False):
        if plain_text:
            with open(path, 'w') as f:
                f.write('\n'.join(self.lexicon.list))
        else:
            with open(path, 'wb') as f:
                import pickle
                pickle.dump(self.lexicon.list, f)

    'Load the lexicon'
    def load_lexicon(self, path, plain_text=False):
        if plain_text:
            with open(path, 'r') as f:
                self.lexicon.list = f.readlines()
        else:
            with open(path, 'rb') as f:
                import pickle
                self.lexicon.list = pickle.load(f)
    
    'Save the segmented corpus'
    def save_segmented_corpus(self, path, segmented_corpus):
        # segmented_corpus is the output of the `apply()` 
        with open(path, 'w') as f:
            if isinstance(segmented_corpus[0], str): # if segmented_corpus has been flattened as a list of tokens
                f.write('|'+'|'.join(segmented_corpus)+'|')
            else: # if segmented_corpus has the sentence level
                f.write('\n'.join(['|'+'|'.join(sent)+'|' for sent in segmented_corpus]))

    'Encode the segmented corpus as a list of integers'
    def encode_segmented_corpus(self, segmented_corpus):
        rev_codebook = dict((token, idx) for idx,token in enumerate(self.lexicon))

        # segmented_corpus is the output of the `apply()` 
        if isinstance(segmented_corpus[0], str): # if segmented_corpus has been flattened as a list of tokens
            encoded = [rev_codebook[token] for token in segmented_corpus]
        else: # if segmented_corpus has the sentence level
            encoded = [[rev_codebook[token] for token in sent] for sent in segmented_corpus]

        codebook = dict((idx,token) for token, idx in rev_codebook.items())
        return codebook, encoded