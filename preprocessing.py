import structures
from dataclasses import dataclass, field
from typing import Dict, Union, Callable
import chardet
import random
import os
import types
import logging
logger = logging.getLogger("")
logger.setLevel(logging.WARNING)

'Extract the elements in corpus and store them to Lexicon.'
def create_lexicon(corpus=[], symbol_only=False):
    # flatten the corpus until the corpus contain no structure higher than sentence
    def flatten(S):
        if S == []:
            return S
        if isinstance(S[0], list):
            return flatten(S[0]) + flatten(S[1:])
        return S[:1] + flatten(S[1:])

    flatted_corpus = flatten(corpus)

    if symbol_only: # extract the symbols in the corupus
        to_feed = sorted(set(''.join(flatted_corpus))) # get the unique symbols in the corpus
    else:
        to_feed = sorted(set(flatted_corpus))

    return structures.TrieList(to_feed) # store the `to_feed` to Lexicon and return the Lexicon

'merge the token lists as strings for each sentence'
def merge_token_list(sents):
    return [''.join(sent) for sent in sents]

'Convert/store the corpus input.'
@dataclass
class Corpus():
    """
    This class can convert (processed by the method `create_corpus`) the raw corpus (in the parameter `train`) 
        to a nested-list corpus: (list of articles ->) list of sentence ->) -> (token)].
    Three types of input are allowed:
        1. the filename of corpus;
        2. the path of corpus files ;
        3. a corpus in string, where `sentence_divider` splits the sentences and `token_divider` splits the tokens (if ); if `self.article_divider` is provided, the parameter splits the articles.
        4. a corpus in the list of sentences:
            a. if `token_divider` is 'None', treat the sentence element as the list of tokens,
            b. otherwise, treat the sentence element as the string with the `token_divider` splitting the tokens;
    """
        
    train: Union[list, str] = None # the training corpus
    test: Union[list, str] = field(default=None, repr=False) # the testing corpus that should have the level of article. if `test` is `None`, the first article (or first 10 sentences) of the training corpus will be used as the testing corpus 
    
    prep: Callable[[str], str] = lambda text: text.strip() # the function to preprocess the sentences
    token_divider: str = None # if None, treating the sentence as a string
    sentence_divider: str = None # the divider of sentences
    article_divider: str = None # if provided, the parameter splits the articles
    remove_token_divider: bool = False # if True, the token dividerd will be removed from the corpus

    has_article_level: bool = False # if the corpus have the article level, it will be set as True automatically
    current_article_position: int = field(default=0, repr=False) # (for iteration) the current article position in the corpus
    current_sentence_position: int = field(default=0, repr=False) # (for iteration) the current sentence position in the article

    note: Dict = None # the note of the corpus

    def __post_init__(self):
        if self.train != None:
            self.train = self.create_corpus(self.train)

            if self.test != None:
                logger.info("Make sure the hierarchy of `test` does not has the article level.")
                self.test = self.create_corpus(self.test)
            else:
                if self.has_article_level:
                    self.test = self.train[0] # the test corpus will be the first article of the training corpus if the training corpus has the article level
                else:
                    self.test = self.train[:10] # the test corpus will be the first 10 sentences of the training corpus if the training corpus do not has the article level

    'Convert the file to the list of sentences.'
    @classmethod
    def file_to_article(cls, path):
        # recognize the encoding of the file by the `chardet` package    
        with open(path, 'rb') as f:
            encoding = chardet.detect(f.readline())['encoding']
        # read the file; treat each file as an article
        with open(path, encoding=encoding) as f: 
            article = f.read().strip()
        return article
    
    'Convert the raw corpus to the LiB-allowed format: corpus -> (article ->) sentence (list of tokens, or a string).'
    def create_corpus(self, raw):
        
        'preprocess the sentence and split the tokens'
        def process_sentence(sent):
            sent = self.prep(sent)
            if isinstance(self.token_divider, str):
                splitter = lambda x: x.split(self.token_divider)
            elif isinstance(self.token_divider, types.FunctionType):
                splitter = lambda x: self.token_divider(x)
            elif self.token_divider == None:
                return sent
            else:
                raise Exception('The parameter "token_divider" should be a str, a function, or None.')

            padding = '' if self.remove_token_divider else self.token_divider
            return [token + padding for token in splitter(self.prep(sent))]
        
        'split the sentences then process them'
        def process_article(art):
            if isinstance(self.sentence_divider, str):
                splitter = lambda x: x.split(self.sentence_divider)
            elif isinstance(self.sentence_divider, types.FunctionType):
                splitter = self.sentence_divider
            else:
                raise Exception('The parameter "sentence_divider" should be a str or a function.')
            
            return [process_sentence(sent) for sent in splitter(art)]

        # fill the `articles` list from the input
        articles = []
        if isinstance(raw, str):
            # if the input is a directory path, treat each file as an article
            if os.path.isdir(raw):
                logger.info("The input of `create_corpus` is a directory path.")
                for fn in os.listdir(raw):
                    article = Corpus.file_to_article(raw+'/'+fn)
                    articles.append(process_article(article))
            # otherwise, the articles should be determined by the `article_divider` parameter
            else:
                # if the input is a file path, load the file
                if os.path.isfile(raw):
                    logger.info("The input of `create_corpus` is a filename.")
                    raw = Corpus.file_to_article(raw)
                # we can ensure that the input is a string-format corpus now

                # if the `article_divider` parameter is provided, split the string into articles
                if self.article_divider:
                    for article in raw.split(self.article_divider):
                        articles.append(process_article(article))
                # otherwise, treat the string as a single article
                else:
                    articles.append(process_article(raw))

        # if the input is a list, treat `raw` as a nested corpus
        elif isinstance(raw, list):
            # if `sentence_divider` is provided, the string should be article
            # treat `raw` as the list of article strings.
            if self.sentence_divider is not None:
                for article in raw:
                    article = process_article(article)
                    articles.append(article)
            # if `sentence_divider` is not provided but `token_divider` is provided, the string should be sentence.
            elif self.token_divider is not None:
                # if the input is a list of sentences, treat `raw` an article
                if isinstance(raw[0], str):
                    article = [process_sentence(sentence) for sentence in raw]
                    articles.append(article)
                # if the input is a 2-level list of sentences, treat `raw` as articles
                elif isinstance(raw[0][0], str):
                    articles = [[process_sentence(sentence) 
                                    for sentence in article] 
                                        for article in raw]
            # if neither `sentence_divider` nor `token_divider` is provided, the string should be token.
            elif self.token_divider is None:
                # if the input is a list of tokens, treat `raw` a sentence
                if isinstance(raw[0], str):
                    articles.append([raw])
                # if the input is a 2-level list of tokens, treat `raw` an article
                elif isinstance(raw[0][0], str):
                    articles.append(raw)
                # if the input is a 3-level list of tokens, treat `raw` as articles
                elif isinstance(raw[0][0][0], str):
                    articles = raw
            else:
                raise Exception('The input list is not a valid corpus.')
        else:
            raise Exception('The parameter "raw" should be a str or a list.')
                    
        if len(articles) > 1:
            self.has_article_level = True
            return articles
        else: # remove the temporal article level added above
            return articles[0]

    'Extract the samples from the training corpus'
    def extract_article(self, length=None, sequential_read=True, loop=True):
        if sequential_read:
            return self.iterate_extract_article(length, loop=loop) # the iteration on the training corpus to extract the sentent samples or an article
        else:
            return self.random_extract_article(length) # the random extraction on the training corpus to extract the sentent samples or an article

    'the random extraction on the training corpus to extract the sentent samples or an article'
    def random_extract_article(self, length=None):
        corpus = self.train

        if self.has_article_level:
            a_ind = random.randint(0, len(corpus))
            self.current_article_position = a_ind
            article = corpus[a_ind]
            if length == None:
                return article
            else:
                if len(article) > length:
                    s_ind = random.randint(0, len(article)-length)
                    self.current_sentence_position = s_ind
                    return article[s_ind: s_ind+length]
                else:
                    self.current_sentence_position = 0
                    return article
        else:
            self.current_article_position = 0
            if length == None:
                s_ind = random.randint(0, len(corpus))
                self.current_sentence_position = s_ind
                return [corpus[s_ind]]
            else:
                if len(corpus) > length:
                    s_ind = random.randint(0, len(corpus)-length)
                    self.current_sentence_position = s_ind
                    return corpus[s_ind: s_ind+length]
                else:
                    self.current_sentence_position = 0
                    return corpus

    'the iteration on the training corpus to extract the sentent samples or an article'
    def iterate_extract_article(self, length=None, loop=True):
        def update_current_article_position():
            if len(self.train) > self.current_article_position + 1:
                self.current_article_position += 1
            else:
                # if the `loop` parameter is `True`, reset the `current_article_position` to `0` and start the next loop
                if loop:
                    self.current_article_position = 0
                    logger.info("Finished the last loop on the training corpus. Ready to start the next loop.")
                # if the `loop` parameter is `False`, raise an exception to stop the iteration
                else:
                    raise Exception('The iteration on training corpus reaches the end.')

        def update_current_sentence_position(delta, N_sent):
            if self.current_sentence_position + delta < N_sent:
                self.current_sentence_position += delta
            else:
                if self.has_article_level:
                    update_current_article_position()
                # if the `loop` parameter is `True`, reset the `current_article_position` to `0` and start the next loop
                if loop:
                    self.current_sentence_position = 0
                    logger.info("Finished the last loop on the training corpus. Ready to start the next loop.")
                # if the `loop` parameter is `False`, raise an exception to stop the iteration
                else:
                    raise Exception('The iteration on training corpus reaches the end.')

        corpus = self.train
        sent_pos, art_pos = self.current_sentence_position, self.current_article_position

        if self.has_article_level:
            if length == None:
                article_ = corpus[art_pos]
                update_current_article_position()
            else:
                article = corpus[art_pos]
                article_ = article[sent_pos: sent_pos+length]
                update_current_sentence_position(len(article_), len(article))
        else:
            if length == None:
                article_ = [corpus[sent_pos]]
                update_current_sentence_position(1, len(corpus))
            else:
                article_ = corpus[sent_pos: sent_pos+length]
                update_current_sentence_position(len(article_), len(corpus))

        return article_

    'Reset the iteration on the training corpus'
    def reset_iterate(self):
        self.current_sentence_position = 0
        self.current_article_position = 0
    