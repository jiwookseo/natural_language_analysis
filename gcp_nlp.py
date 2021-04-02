#!/usr/bin/env python
# coding: utf-8

# Imports the Google Cloud client library
from google.cloud import language_v1
from google.protobuf import json_format
import json
import pandas as pd
import numpy as np


# Instantiates a client
client = language_v1.LanguageServiceClient()
    
class Tda:
    def __init__(self, text=None, path=None):
        # google language document & response
        if text:
            self.doc = language_v1.Document({"content": text, "type_": "PLAIN_TEXT"})
            self.api_response = self.get_response()
            # result json, dictionary
            self.json = self.api_response.__class__.to_json(self.api_response) 
            self.result = json.loads(self.json)
        # load existing result
        else:
            with open(f"data/{0}.json", "r", encoding="UTF-8") as f:
                self.result = json.load(f)
        
        # extract pandas Dataframe
        self.sentences = self.get_sentences()
        self.tokens = self.get_tokens()
        self.entities = self.get_entities()
        
        # document sentiment, type => {'score': float, 'magnitude': float}
        self.sentiment = self.result["documentSentiment"]
        
        # category list, but empty
        self.categories = self.result["categories"]
    
    # analyze requestm syntax / entities / document_sentiment
    def get_response(self):
        return client.annotate_text(
            request={"document": self.doc,
                     "encoding_type": "UTF8",
                     # korean language is not supported for "extract_entity_sentiment", "classify_text"
                     "features": {
                         "extract_syntax": True,
                         "extract_entities": True,
                         "extract_document_sentiment": True}})
    
    def get_sentences(self):
        sentences = self.result["sentences"]
        if not sentences:
            return None
        sentences_flatten = [(
            sentence["text"]["content"],
            sentence["sentiment"]["score"],
            sentence["sentiment"]["magnitude"]
        ) for sentence in sentences]
        # magnitude = abs(sentiment)
        return pd.DataFrame(sentences_flatten, columns=["content", "sentiment", "magnitude"])


    def get_tokens(self):
        tokens = self.result["tokens"]
        if not tokens:
            return None
        tokens_flatten = [(
            token["text"]["content"],
            token["lemma"]
        ) for token in tokens]
        return pd.DataFrame(tokens_flatten, columns=["token", "lemma"])


    def get_entities(self):
        entities = self.result['entities']
        entities_flatten = [(
            entity["name"],
            entity["salience"]
        ) for entity in entities]
        entities_df = pd.DataFrame(entities_flatten, columns=["name", "salience"])
        df_sum = entities_df.groupby(['name']).sum()
        df_count = entities_df.groupby(['name']).count()
        df_count.rename(columns = {'salience' : 'count'}, inplace = True)
        return pd.concat([df_sum, df_count], axis=1).sort_values(by=['salience'], ascending=False)