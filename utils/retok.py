#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

import retok
t = retok.ReTokenizer('bert-base-uncased')
e, ind, l = t.retokenize_and_encode_indexed('hellooooo brave new world !!!!'.split())
e, ind, l = t.retokenize_and_encode_indexed([str(i) for i in range(600)])

@author: remstef
"""

from transformers import AutoTokenizer

class ReTokenizer(object):

  def __init__(self, transformer_modelname):
      super(ReTokenizer,self).__init__()
      self.modeltokenizer = AutoTokenizer.from_pretrained(transformer_modelname, use_fast=False)

  def retokenize_and_encode(self, tokens):
    # retokenize sentence to match transformers model
    sentence_tok_model = [ ]
    token2modeltoken = [ ]
    modeltoken2token = [ ]
    for ti, token in enumerate(tokens):
      token2modeltoken.append([ ])
      for modeltoken in self.modeltokenizer.tokenize(token.strip()):
        token2modeltoken[ti].append(len(sentence_tok_model))
        sentence_tok_model.append(modeltoken.strip())
        modeltoken2token.append(ti)
    # encode tokens
    encoded_input = self.modeltokenizer.encode_plus(sentence_tok_model, padding='max_length', truncation=True, max_length=min(self.modeltokenizer.max_len_single_sentence, len(sentence_tok_model)+2), return_tensors='pt')
    sentence_tok_model_decoded = self.modeltokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
    # realign
    modeltoken2realignedmodeltoken = [ ]
    shift = 0
    for i, modeltoken in enumerate(sentence_tok_model):
      while (i+shift < len(sentence_tok_model_decoded)) and modeltoken != sentence_tok_model_decoded[i+shift]:
        shift += 1
      if i+shift >= len(sentence_tok_model_decoded):
        break
      modeltoken2realignedmodeltoken.append(i+shift)
    # ===
    return encoded_input, sentence_tok_model, sentence_tok_model_decoded, token2modeltoken, modeltoken2token, modeltoken2realignedmodeltoken

  def retokenize_and_encode_indexed(self, tokens):
    # retokenize
    encoded_input, sentence_tok_model, sentence_tok_model_decoded, token2modeltoken, modeltoken2token, modeltoken2realignedmodeltoken = self.retokenize_and_encode(tokens)
    # prepare forward and reverse indices
    # i --> (token, [encoded_indices], [decoded_tokens])
    tok_fw_index = { i: ( ti, [ modeltoken2realignedmodeltoken[j] if j < len(modeltoken2realignedmodeltoken) else -1 for j in token2modeltoken[i] ], [ sentence_tok_model_decoded[modeltoken2realignedmodeltoken[j]] if j < len(modeltoken2realignedmodeltoken) else None for j in token2modeltoken[i] ] )for i, ti in enumerate(tokens) }
    # i --> (modeltoken, tokenindex, modeltoken)
    tok_rev_index = { i: (modeltoken2token[modeltoken2realignedmodeltoken.index(i)] if i in modeltoken2realignedmodeltoken else -1, mt) for i, mt in enumerate(sentence_tok_model_decoded) }
    # ===
    return encoded_input, (tok_fw_index, tok_rev_index), (sentence_tok_model, sentence_tok_model_decoded, token2modeltoken, modeltoken2token, modeltoken2realignedmodeltoken)
