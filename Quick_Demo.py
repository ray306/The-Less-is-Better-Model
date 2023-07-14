import LiB

'''-----------Load corpus-----------'''
'The input is a text file with each sentence separated by a newline and each token separated by a space.'
corpus = LiB.Corpus('corpus/br-text.txt', sentence_divider='\n', token_divider=' ')
# corpus = LiB.Corpus('zh_sent.txt', token_divider=' ', sentence_divider='\n', remove_token_divider=True)
# corpus = LiB.Corpus('ctb.txt', token_divider=' ', sentence_divider='\n', remove_token_divider=True)

'The input is a directory path with each file as an article, each sentence separated by a newline, and each token separated by a space.'
# corpus = LiB.Corpus('corpus', sentence_divider='\n', token_divider=' ')

'The input is a string with each sentence separated by a newline and each token separated by a space.'
# s = "you want to see the book\ncan you feed it to the doggie\nwhat's it\nget it"
# corpus = LiB.Corpus(s, sentence_divider='\n', token_divider=' ')

""" The corpus_ is a list of sentences, each sentence is a list of tokens.
If the token_divider is not None, the corpus_ should be a list of strings, each string is a sentence with tokens separated by the token_divider.
If the sentence_divider is not None, the corpus_ should be a string, with sentences separated by the sentence_divider. """;
# with open('sents_with_line_ends_without_repeat_en', 'rb') as f:
#     import pickle
#     corpus_ = pickle.load(f)
# corpus = LiB.Corpus(corpus_, token_divider=None)


'''-----------Run model-----------'''
'Init a model with parameters (explained in the code script)'
m = LiB.Model(corpus, in_rate=0.25, out_rate=0.01, update_rate=0.1, life=3, 
              inspect_context=True)

'RUN!!!'
# The model learns 200 steps
for step_id in range(0,501):
    # The model learns an article with 200 tokens in each step, and reports the metrics every 100 steps
    m.learn(step_id, article_length_max=200, report_interval=100);
m.clean_lexicon()
            
'Report'
print(f"Time cost for training: {sum(m.logs['time_cost'])//60} min {sum(m.logs['time_cost'])%60:.1f}s\n")

print('Orignal/LiB segmentation:')
m.segment_and_show()


'''-----------Misc-----------'''
# m.save_lexicon('lexicon.txt', plain_text=False)
# m.load_lexicon('lexicon.txt', plain_text=False)

# segmented_corpus = m.apply(m.corpus.test, flatten=False)

# m.save_segmented_corpus(path, segmented_corpus)
# codebook, encoded = m.encode_segmented_corpus(segmented_corpus)


# import math
# # The Huffman Coding algorithm comes from https://github.com/arnab132/Huffman-Coding-Python/blob/master/huffman.py
# def huffman_coding(corpus, show_codebook=False):
#     # Creating tree nodes
#     class NodeTree(object):
#         def __init__(self, left=None, right=None):
#             self.left = left
#             self.right = right

#         def children(self):
#             return (self.left, self.right)

#         def nodes(self):
#             return (self.left, self.right)

#         def __str__(self):
#             return '%s_%s' % (self.left, self.right)

#     # Main function implementing huffman coding
#     def huffman_code_tree(node, left=True, binString=''):
#         if type(node) is str:
#             return {node: binString}
#         (l, r) = node.children()
#         d = dict()
#         d.update(huffman_code_tree(l, True, binString + '0'))
#         d.update(huffman_code_tree(r, False, binString + '1'))
#         return d

#     # Calculating frequency
#     freq = {}
#     for c in corpus:
#         if c in freq:
#             freq[c] += 1
#         else:
#             freq[c] = 1

#     freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

#     nodes = freq

#     while len(nodes) > 1:
#         (key1, c1) = nodes[-1]
#         (key2, c2) = nodes[-2]
#         nodes = nodes[:-2]
#         node = NodeTree(key1, key2)
#         nodes.append((node, c1 + c2))

#         nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

#     huffman_code = huffman_code_tree(nodes[0][0])
#     if show_codebook:
#         print("[Huffman Encoding]:")
#         for token, code in huffman_code.items():
#             print(f"{token}:\t{code}")

#     return huffman_code, nodes[0][0]

# def calculate_MDL(segmented_corpus, tree, encoding, show_encoded_corpus=False):
#     # serialize the Huffman tree to a binary string with the labels of the leafs of the tree, by depth-first search
#     def serialize_dfs(node, label='', tree='', values=[]):
#         tree += label
#         if type(node) is str:
#             tree += '#'
#             values.append(node)
#             return tree, values
#         else:
#             (l, r) = node.children()
#             tree, values = serialize_dfs(l, '0', tree, values)
#             tree, values = serialize_dfs(r, '1', tree, values)
#         return tree, values

#     serialized_tree, serialized_values = serialize_dfs(tree) # serialize the tree to a binary string ('0/1') with the label of its leaves ('#'), and a list of to store the values of the leaves.
#     # the number of bits to store the tree structure
#     tree_bits = math.ceil(math.log2(3**len(serialized_tree))) # the string is trinary (0/1/#), it would be converted to binary to calculate the number of bits
#     # the number of bits to store the values of the tree leaves (the dictionary)
#     values_bits = len('\t'.join(serialized_values).encode('utf-8')) * 8 # the list would be represented by a string that is separated by '\t', and then encoded to utf-8 to calculate the number of bits
#     model_bits = tree_bits + values_bits

#     encoded_corpus = ""
#     for token in segmented_corpus:
#         encoded_corpus += encoding[token]
#     encoded_bits = len(encoded_corpus) # the number of bits to store the encoded corpus
#     if show_encoded_corpus:
#         print("[Encoded Corpus]:")
#         print(encoded_corpus)

#     mdl = model_bits + encoded_bits
#     return model_bits, encoded_bits, mdl

# flat_segmented_corpus = m.apply(m.corpus.train, flatten=True)
# huffman_code, huffman_tree = huffman_coding(flat_segmented_corpus, show_codebook=True)
# dictionary_bits, encoded_bits, mdl = calculate_MDL(flat_segmented_corpus, 
#                                                    huffman_tree, huffman_code, 
#                                                    show_encoded_corpus=False)
# print(f'Codebook:\t{dictionary_bits} bits\nEncoded_corpus:\t{encoded_bits} bits\nMDL:\t\t{mdl} bits')