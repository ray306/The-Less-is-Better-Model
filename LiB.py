import random
import time
import math
import importlib
import gc

from collections import Counter
from itertools import accumulate,groupby
from statistics import mean
from typing import Iterator, Dict, List, Tuple, Union
from dataclasses import dataclass, field

import structures

@dataclass
class model():
    corpus_train: List = field(repr=False)
    corpus_test: List = field(default=None, repr=False)
    lexicon: Union[structures.TrieList, list, None] = field(default=None, repr=False)

    life: int = 10
    lexicon_in: float = 0.25
    lexicon_out: float = 0.0001
    update_rate: float = 0.2
    max_len: Union[None, int] = None
    skip_gap: Union[None, int] = None
    largest_only: bool = False
    
    probation: Dict = field(default_factory=dict, repr=False)
    logs: Dict = field(default_factory=dict, repr=False)

    random_seed: int = 0

    def __post_init__(self):
        random.seed(self.random_seed)
        self.probation = dict()
        self.to_mem = set()
        self.logs = {'reward_lists': [], 'eval_index': [], 'time_cost': [], 'note': None}

        if self.corpus_test is None:
            self.corpus_test = self.corpus_train
        if self.lexicon is None:
            self.lexicon = model.create_lexicon()
            self.preload(self.corpus_train[0])
        elif isinstance(self.lexicon, list):
            self.lexicon = model.create_lexicon(self.lexicon)
        else:
            self.lexicon = self.lexicon.copy() # avoid to modify the original lexicon

        self.corpus_test = model.extract_article(self.corpus_test, length=None, with_joined=True)

    def preload(self, corpus_samples):
        def flatten(S):
            if S == []:
                return S
            if isinstance(S[0], list):
                return flatten(S[0]) + flatten(S[1:])
            return S[:1] + flatten(S[1:])

        self.probation.clear()
        for i in sorted(set(''.join(flatten(corpus_samples)))):
            self.lexicon.append(i)
            self.probation[i] = self.life

    def evaluate(self, to_test, doc, doc_covered, doc_loc, reward_list):
        for _chunk_ in list(to_test.keys()):

            onset, end, N_unknowns, N_chunks, chunks_list_0, chunks_list = to_test[_chunk_]
            while end < doc_loc:
                if self.max_len == None:
                    slice_end = None
                else:
                    slice_end = end+self.max_len

                chunk_next = self.lexicon.match(doc[end: slice_end])
                chunks_list.append(chunk_next)
                chunk_size_next = len(chunk_next)
                if chunk_size_next > 0:
                    end += chunk_size_next
                    N_chunks += 1
                else:
                    end += 1
                    N_unknowns += 1
                    N_chunks += 1
                    
            if end == doc_loc:
                del to_test[_chunk_]

                doc_section = doc_covered[onset: end]

                N_chunks_0 = len(chunks_list_0)
                
                redundant = (N_chunks_0 > N_chunks) or \
                               (N_chunks_0 == N_chunks and \
                                    sum(self.lexicon.index_with_prior(i) for i in chunks_list_0 if i in self.lexicon) > sum(self.lexicon.index_with_prior(i) for i in chunks_list if i in self.lexicon))

                if redundant:
                    smaller_chunk = self.lexicon.match(_chunk_[:-1])
                    reward_list.append((_chunk_, 1))
                    reward_list.append((smaller_chunk, -1))
                else:
                    reward_list.append((_chunk_, -1))

                    if _chunk_ in self.probation:
                        del self.probation[_chunk_]
            else:
                to_test[_chunk_] = onset, end, N_unknowns, N_chunks, chunks_list_0, chunks_list
            
    def batch_update_lexicon(self, reward_list):
        # for _chunk_, label in reward_list:
        #     if _chunk_ in self.lexicon.relationship:
        #         reward_list.append((self.lexicon.relationship[_chunk_], label))

        self.logs['reward_lists'].append(reward_list)

        forget_list = []
        for _chunk_ in list(self.probation.keys()):
            current_life = self.probation[_chunk_]
            if current_life <= 1:
                del self.probation[_chunk_]
                forget_list.append(_chunk_)
            else:
                self.probation[_chunk_] = current_life - 1

        self.lexicon.group_remove(forget_list)
        reward_list = [(w,e) for w,e in reward_list if w not in set(forget_list)]
        self.lexicon.group_move(reward_list, self.update_rate, pre_update=False)

        for ind in range(int((1-self.lexicon_out)*len(self.lexicon)), len(self.lexicon)):
            _chunk_ = self.lexicon[ind]
            if _chunk_ not in self.probation:
                self.probation[_chunk_] = self.life

    def memorize(self, chunk):
        if chunk and not self.lexicon.search(chunk):
            self.lexicon.append(chunk)
            self.probation[chunk] = self.life
            # if chunk in self.to_mem:
            #     self.lexicon.append(chunk)
            #     self.probation[chunk] = self.life
            # else:
            #     self.to_mem.add(chunk)

    def reading(self, article):
        used_chunks = Counter()
        self.to_mem = set()

        doc_covered_all = []
        reward_list = []
        l_lexicon_size = len(self.lexicon)

        segmented = []

        for sent in article:
            chunks_in_sent = []
            singles = []
            to_test = dict()
            last_chunk = ['', 0] # 0 unknown, 1 known
            
            sent_covered = [0] * len(sent)
            i = 0
            while i < len(sent):
                self.evaluate(to_test, sent, sent_covered, i, reward_list)

                if self.max_len == None:
                    slice_end = None
                else:
                    slice_end = i+self.max_len

                chunk_2nd, chunk = self.lexicon.match_two(sent[i: slice_end])
                
                for _chunk_ in list(to_test.keys()):
                    to_test[_chunk_][4].append(chunk)
                if len(chunk) > 0:
                    if (random.random() < self.lexicon_in):
                        to_get = None
                        if last_chunk[1] == 0: 
                            to_get = last_chunk[0]
                        else:
                            to_get = last_chunk[0] + chunk

                        self.memorize(to_get)

                    if len(chunk) > 1 and len(chunk_2nd) > 0:
                        to_test[chunk] = [i, i + len(chunk_2nd), 0, 1, [chunk], [chunk_2nd]] # _chunk_, start position, current position, number of unknowns, number of chunks
                    elif chunk in self.probation:
                        del self.probation[chunk]

                    chunk_s = len(chunk)
                    sent_covered[i: i + chunk_s] = [sent_covered[i-1] + 1] * chunk_s
                    chunks_in_sent.append(chunk)
                    i += chunk_s
                    last_chunk = [chunk, 1]
                    used_chunks[chunk] += 1
                    
                else:
                    if last_chunk[1] == 1:
                        last_chunk = ['', 0]
                    last_chunk[0] += sent[i]
                    chunks_in_sent.append(sent[i])
                    i += 1

    #         if use_skip:
    #             chunks_in_sent = ['[bos]'] + chunks_in_sent + ['[eos]']
    #             for a,b in zip(chunks_in_sent[:-2], chunks_in_sent[2:]):
    #                 if random.random() < self.lexicon_in_sk and not (a=='[bos]' and b=='[eos]'):
    #                     self.memorize(('skipgram', a,b), reward_list)
                        
    #             while 1:
    #                 skip_gram, skip, chunks_in_sent = self.lexicon.skipgram_match(chunks_in_sent)    
    #                 if skip is not None and len(skip) > 1:
    #                     skip = ''.join(skip)
    # #                     if (len(skip) < 8):print(skip)
    #                     if (len(skip) <= mini_gap) and (random.random() < self.lexicon_in):
    #     #                     self.memorize(skip, reward_list)
    #                         if not self.lexicon.search(skip):
    #                             self.lexicon.relationship[skip] = ('skipgram', *skip_gram)
    #                             self.memorize(skip, reward_list)
    #                             # self.lexicon.append(skip)
    #                 if chunks_in_sent == None:
    #                     break

            doc_covered_all += sent_covered

            segmented.append(chunks_in_sent)

        self.batch_update_lexicon(reward_list)

        in_count = len(self.lexicon) - l_lexicon_size
        covered_rate, chunk_groups = (1-sum([i==0 for i in doc_covered_all])/len(doc_covered_all), 
                [(key, len(list(group))) for key, group in groupby(doc_covered_all)])
        chunk_in_use = sum([g_len if key==0 else 1 for key, g_len in chunk_groups])
        mem_usage = len(set(used_chunks.keys()) & set(self.lexicon))/len(self.lexicon)
        
        return covered_rate, len(doc_covered_all)/chunk_in_use, mem_usage, in_count, segmented

    def run(self, epoch_id, article_length=500, test_interval=100):
        # if epoch_id==0:
        #     article, article_raw = self.corpus_test
        #     to_test = self.show_reading(article,return_chunks=True)
        #     avg_chunk_len = sum(len(w) for w in to_test)/len(to_test)
        #     codebook = ''.join(self.lexicon.par_list)
        #     chunk_counter = Counter(to_test)
        #     char_counter = Counter(codebook)
        #     LM = -sum(char_counter[t]*math.log2(char_counter[t]/len(codebook)) for t in char_counter)
        #     LD = -sum(chunk_counter[t]*math.log2(chunk_counter[t]/len(to_test)) for t in chunk_counter)

        #     print(f'{epoch_id}\t  MemLength: {len(self.lexicon)}\t  ChunkLength: {avg_chunk_len:.2f}\t  B: {LM:.0f} {LD:.0f} {LM+LD:.0f} {LM*LD/100000000000:.0f}')
        #     print()

        start_time = time.time()
        article, article_raw = model.extract_article(self.corpus_train, length=article_length, with_joined=True)
        covered_rate, avg_chunk_len, mem_usage, in_count, segmented = self.reading(article)
        self.logs['time_cost'].append(time.time()-start_time)

        if epoch_id % test_interval == 0:  
            article, article_raw = self.corpus_test
            to_test = self.show_reading(article,return_chunks=True)
            avg_chunk_len = sum(len(w) for w in to_test)/len(to_test)
            codebook = ''.join(list(set(to_test)))
            chunk_counter = Counter(to_test)
            char_counter = Counter(codebook)
            LM = -sum(char_counter[t]*math.log2(char_counter[t]/len(codebook)) for t in char_counter)
            LD = -sum(chunk_counter[t]*math.log2(chunk_counter[t]/len(to_test)) for t in chunk_counter)

            eval_index = math.log2(len(self.lexicon))/avg_chunk_len
            # math.log10(len(self.lexicon))/avg_chunk_len
            # print(math.log2(len(self.lexicon)),len(self.lexicon),avg_chunk_len,len(self.lexicon)/avg_chunk_len)

            self.logs['note'] = avg_chunk_len

            self.logs['eval_index'].append(eval_index)

            mem_avg = mean([len(i) for i in self.lexicon.par_list])
              
            # print(f'{epoch_id}\t  MemLength: {len(self.lexicon)}\t  ChunkLength: {avg_chunk_len:.2f}\t  B: {LM:.0f} {LD:.0f} {LM+LD:.0f} {self.lexicon_out*self.life*LM+self.lexicon_in*LD:.0f} {self.lexicon_out*self.life*math.log(len(self.lexicon))+self.lexicon_in*avg_chunk_len:.3f} {LM*LD/1000000000:.0f} {math.log(len(self.lexicon))/avg_chunk_len:.3f}')
            if epoch_id==0:
                print(f'{epoch_id}\t  MemLength: {len(self.lexicon)} {mem_avg:.1f}\t  ChunkLength: {avg_chunk_len:.2f}\t  EvalIndex: {eval_index:.3f}')
            else:
                print(f'{epoch_id}\t  MemLength: {len(self.lexicon)} {mem_avg:.1f}\t  ChunkLength: {avg_chunk_len:.2f}\t  EvalIndex: {eval_index:.3f} Change:{(self.logs["eval_index"][-2]-self.logs["eval_index"][-1])/self.logs["eval_index"][-2]*100:.1f}%')
            print()

            # article, article_raw = self.corpus_test
            # chunk_pos = self.show_reading(article)
            # avg_chunk_len = sum(len(sent) for sent in article)/len(chunk_pos)

            # self.logs['eval_index'].append(math.log(len(self.lexicon))/avg_chunk_len)
              
            # print(f'{epoch_id}\t  MemLength: {len(self.lexicon)}\t  ChunkLength: {avg_chunk_len:.2f}\t  B: {math.log(len(self.lexicon))/avg_chunk_len:.3f}')
            # print()

            # article, article_raw = model.extract_article(self.corpus_test, with_joined=True)
            # chunk_pos_0 = set(accumulate([len(c) for sent in article_raw for c in sent]))

            # chunk_pos = self.show_reading(article, decompose=True)

            # precision_1 = len(chunk_pos_0&chunk_pos)/len(chunk_pos)
            # recall_1 = len(chunk_pos_0&chunk_pos)/len(chunk_pos_0)
            
            # chunks_0 = [[c for c in sent] for sent in article_raw]

            # chunks = self.show_reading(article, return_chunks=True, comb_sents=False, decompose=True)

            # precision_2, recall_2, f1_2 = model.get_f1(chunks_0, chunks)

            # avg_chunk_len = sum(len(sent) for sent in article)/len(chunk_pos)
            # print(f'{epoch_id}\t  MemLength: {len(self.lexicon)}\t  ChunkLength: {avg_chunk_len:.2f}\t  B: {math.log(len(self.lexicon))/avg_chunk_len:.3f}')
            # print(f'[B] Precision: {precision_1*100:.2f}% \t Recall: {recall_1*100:.2f}% \t F1: {2*precision_1*recall_1/(precision_1+recall_1)*100:.2f}%')
            # print(f'[L] Precision: {precision_2*100:.2f}% \t Recall: {recall_2*100:.2f}% \t F1: {2*precision_2*recall_2/(precision_2+recall_2)*100:.2f}%')
            # print()

            # print(int((1-self.lexicon_out)*len(self.lexicon)), len(self.lexicon),len(self.probation))
            # errors = model.count_error(self.lexicon, lexcion, chunks_0, chunks,lexicon)
            # errors_all.extend(errors)

            # gc.collect()
            # return Counter(errors).most_common()

        return segmented
    

    def find_subs(self, large_chunk, level=2):
        chunk_1 = large_chunk

        subs = []
        while 1:
            chunk_1 = self.lexicon.match(chunk_1[:-1])
            chunk_2 = large_chunk[len(chunk_1):]
            
            if chunk_1!='' and chunk_2 in self.lexicon:
                subs.append((chunk_1, chunk_2, (self.lexicon.index_with_prior(chunk_1), self.lexicon.index_with_prior(chunk_2))))
            if len(chunk_1) <= 1:
                break
        
        if len(subs) > 0:
            sub = sorted(subs, key=lambda x:x[2])[0]
            if max(sub[2]) < self.lexicon.index_with_prior(large_chunk, nothing=len(self.lexicon)):
                if level == 1:
                    return sub[:2]
                else:
                    return self.find_subs(sub[0], level-1) + self.find_subs(sub[1], level-1)
            else:
                return (large_chunk,)
        else:
            return (large_chunk,)
        
    def show_reading(self, article, decompose=False, display=False, return_chunks=False, comb_sents=True):
        chunks = []
        
        for sent in article:
            sent_chunks = []
            i = 0

            while i < len(sent):
                if self.max_len == None:
                    slice_end = None
                else:
                    slice_end = i+self.max_len

                if self.largest_only:
                    chunk = self.lexicon.match(sent[i: slice_end])
                    chunk_2nd = ''
                else:
                    chunk_2nd, chunk = self.lexicon.match_two(sent[i: slice_end])

                if len(chunk) > 0:
                    if len(chunk) > 1 and len(chunk_2nd) > 0:
                        onset, end_0, end = i, i + len(chunk), i + len(chunk_2nd)
                        N_unknowns_0, N_chunks_0, N_unknowns, N_chunks =  0, 1, 0, 1
                        chunks_t_0 = [chunk]
                        chunks_t = [chunk_2nd]

                        while end_0 != end:
                            if end_0 > end:
                                next_chunk = self.lexicon.match(sent[end:end+10])
                                chunk_size_next = len(next_chunk)
                                if chunk_size_next > 0:
                                    end += chunk_size_next
                                    N_chunks += 1
                                    chunks_t.append(next_chunk)
                                else:
                                    end += 1
                                    N_unknowns += 1
                                    N_chunks += 1
                                    chunks_t.append(sent[end-1])

                            elif end_0 < end:
                                next_chunk = self.lexicon.match(sent[end_0:end+10])
                                chunk_size_next = len(next_chunk)
                                if chunk_size_next > 0:
                                    end_0 += chunk_size_next
                                    N_chunks_0 += 1
                                    chunks_t_0.append(next_chunk)

                                else:
                                    end_0 += 1
                                    N_unknowns_0 += 1
                                    N_chunks_0 += 1
                                    chunks_t_0.append(sent[end_0-1])

                        redundant = N_unknowns_0 == N_unknowns and \
                                  ((N_chunks_0 > N_chunks) or \
                                   (N_chunks_0 == N_chunks and \
                                        sum(self.lexicon.index_with_prior(i, nothing=len(self.lexicon)) for i in chunks_t_0) > \
                                        sum(self.lexicon.index_with_prior(i, nothing=len(self.lexicon)) for i in chunks_t)))

                        if N_unknowns_0 > N_unknowns or redundant:
                            sent_chunks += chunks_t
                            i += sum(len(c) for c in chunks_t)
                        else:
                            sent_chunks += chunks_t_0
                            i += sum(len(c) for c in chunks_t_0)

                    else:
                        sent_chunks.append(chunk)
                        i += len(chunk)
                else:
                    i += 1
                    sent_chunks.append(sent[i-1]) 
        
            chunks.append(sent_chunks)
            
        if decompose:
            chunks = [[sub_c for c in sent for sub_c in self.find_subs(c)] for sent in chunks]
            
        if comb_sents:
            chunks = [c for sent in chunks for c in sent]
            
        if display:
            if comb_sents:
                print(' '.join(chunks))
            else:
                for sent in chunks: print(' '.join(sent))
            
        if return_chunks:
            return chunks
        else:  
            return set(accumulate([len(c) for c in chunks]))

    def show_result(self, article_raw, decompose=False):
        article, article_raw = self.extract_article(article_raw, with_joined=True)
        for sent, sent_raw in zip(article, article_raw):
            print('|'.join(sent_raw))
            print('|'.join(self.show_reading([sent],return_chunks=True)))
            print()
                
    def demo(self, article_raw, decompose=False, section=(0,-1)):
        onset, end = section
        count = 0
        for chunk_i in range(999):
            if count == len(''.join(article[:onset])):
                chunk_i_0 = chunk_i
            elif count == len(''.join(article[:end])):
                break

            count += len(article_raw[chunk_i])

        self.show_result(article_raw[chunk_i_0: chunk_i], decompose=decompose)

    @classmethod
    def create_corpus(cls, raw_corpus):  
        corpus = []
        if isinstance(raw_corpus, list):
            for l in raw_corpus:
                corpus.append([w+' ' for w in l.replace('\n','').split(' ')])
        elif isinstance(raw_corpus, str):
            if raw_corpus.endswith('.txt'):
                with open(raw_corpus) as f: 
                    for l in f.readlines():
                        corpus.append([w+' ' for w in l.replace('\n','').split(' ')])
            else:
                for l in raw_corpus.split('\n'):
                    corpus.append([w+' ' for w in l.split(' ')])

        else:
            raise Exception('The parameter "raw_corpus" should be a str or a list.')

        return corpus
                
    @classmethod
    def create_lexicon(cls, lexicon_content=[]):  
        importlib.reload(structures)   
        return structures.TrieList(lexicon_content)

    @classmethod
    def extract_article(cls, corpus, length=None, with_joined=False):
        if length == None:
            if isinstance(corpus[0][0], str):
                # print('The corpus contains only one document and the parameter "length" is unknown.')
                article_raw = corpus
            else:
                article_raw = random.choice(corpus)
        else:
            if isinstance(corpus, Iterator):
                article_raw = []
                for i in range(length):
                    try:
                        article_raw.append(next(corpus))
                    except:
                        if len(article_raw) == 0:
                            raise Exception("No more material in this corpus.")

            elif isinstance(corpus[0][0], list):
                article_raw_t = random.choice(corpus)
                if len(article_raw_t) > length:
                    s_ind = random.randint(0, len(article_raw_t)-length)
                    article_raw = article_raw_t[s_ind: s_ind+length]
                else:
                    article_raw = article_raw_t
            else:
                if len(corpus) > length:
                    # article_raw = [random.choice(corpus) for i in range(length)]
                    s_ind = random.randint(0, len(corpus)-length)
                    article_raw = corpus[s_ind: s_ind+length]
                else:
                    article_raw = corpus

        if with_joined:
            article = [''.join(sent) for sent in article_raw]
            return article, article_raw
        else:
            return article_raw

    @classmethod
    def get_f1(cls, gold, est):
        gold_chunk = est_chunk = correct_chunk = 0
        for goldSent,estSent in zip(gold, est):
            gold_chunk += len(goldSent)
            est_chunk += len(estSent)
            goldChunkSet = set(goldSent)
            correct_chunk += sum(im in goldChunkSet for im in estSent)

        pre = correct_chunk / est_chunk
        rec = correct_chunk / gold_chunk
        
        if pre + rec==0:
            return 0, 0, 0
        else:
            f1 = 2 * pre * rec / (pre + rec)
            return pre, rec, f1

    @classmethod
    def count_error(cls, lexcion, chunks_0, chunks, lexicon=None):
        errors = [[[''],['']]]
        for a,b in zip(chunks_0, chunks):
            i, j = 0, 0
            a_len, b_len = 0, 0
            while i<len(a) and j < len(b):
                if a[i] == b[j]:
                    i += 1
                    j += 1 
                else:
                    while 1:
                        a_len = len(''.join(errors[-1][0]))
                        b_len = len(''.join(errors[-1][1]))
                        if a_len == b_len:
                            errors[-1][0].append(a[i])
                            errors[-1][1].append(b[j])
                        elif a_len < b_len:
                            errors[-1][0].append(a[i])
                        elif a_len > b_len:
                            errors[-1][1].append(b[j])

                        a_len = len(''.join(errors[-1][0]))
                        b_len = len(''.join(errors[-1][1]))
                        if a_len == b_len:
                            errors.append([[''],['']])
                            i += 1
                            j += 1   
                            break
                        elif a_len < b_len:
                            i += 1
                        elif a_len > b_len:
                            j += 1 

        if lexicon is None:
            return [(tuple(i1[1:]),tuple(i2[1:])) for i1,i2 in errors[:-1]]
        else:
            def index(i):
                if i in lexicon:
                    return (i,lexicon.index(i))
                else:
                    return (i,'NA')
            return [(tuple(index(i) for i in i1[1:]),tuple(index(i) for i in i2[1:])) for i1,i2 in errors[:-1]]
