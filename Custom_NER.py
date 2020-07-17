import pandas as pd
import json
import logging
import sys
import plac
import logging
import argparse
import os
import pickle
import spacy
import en_core_web_sm
from __future__ import unicode_literals, print_function
import random
from pathlib import Path
from spacy.util import minibatch, compounding
import re


data = pd.read_csv("C:/Users/7325598/Desktop/tensorflow practice/1014_4361_bundle_archive/ner_dataset.csv", encoding= 'unicode_escape')

data = data[['Word','Tag']]
data.to_csv("C:/Users/7325598/Desktop/tensorflow practice/1014_4361_bundle_archive/ner_dataset_new.tsv",sep='\t', index=False)


def tsv_to_json_format(input_path,output_path,unknown_label):
    try:
        f=open(input_path,'r') # input file
        fp=open(output_path, 'w') # output file
        data_dict={}
        annotations =[]
        label_dict={}
        s=''
        start=0
        for line in f:
            if line[0:len(line)-1]!='.\tO':
                word,entity=line.split('\t')
                s+=word+" "
                entity=entity[:len(entity)-1]
                if entity!=unknown_label:
                    if len(entity) != 1:
                        d={}
                        d['text']=word
                        d['start']=start
                        d['end']=start+len(word)-1  
                        try:
                            label_dict[entity].append(d)
                        except:
                            label_dict[entity]=[]
                            label_dict[entity].append(d) 
                start+=len(word)+1
            else:
                data_dict['content']=s
                s=''
                label_list=[]
                for ents in list(label_dict.keys()):
                    for i in range(len(label_dict[ents])):
                        if(label_dict[ents][i]['text']!=''):
                            l=[ents,label_dict[ents][i]]
                            for j in range(i+1,len(label_dict[ents])): 
                                if(label_dict[ents][i]['text']==label_dict[ents][j]['text']):  
                                    di={}
                                    di['start']=label_dict[ents][j]['start']
                                    di['end']=label_dict[ents][j]['end']
                                    di['text']=label_dict[ents][i]['text']
                                    l.append(di)
                                    label_dict[ents][j]['text']=''
                            label_list.append(l)                          
                            
                for entities in label_list:
                    label={}
                    label['label']=[entities[0]]
                    label['points']=entities[1:]
                    annotations.append(label)
                data_dict['annotation']=annotations
                annotations=[]
                json.dump(data_dict, fp)
                fp.write('\n')
                data_dict={}
                start=0
                label_dict={}
    except Exception as e:
        logging.exception("Unable to process file" + "\n" + "error = " + str(e))
        return None

tsv_to_json_format("C:/Users/7325598/Desktop/tensorflow practice/1014_4361_bundle_archive/ner_dataset_new.tsv",'C:/Users/7325598/Desktop/tensorflow practice/1014_4361_bundle_archive/ner_dataset_new.json','abc')


input_file = 'C:/Users/7325598/Desktop/tensorflow practice/1014_4361_bundle_archive/ner_dataset_new.json'

# preparing training data
training_data = []
lines=[]
LABELS = []
with open(input_file, 'r') as f:
    lines = f.readlines()

for line in lines:
    data = json.loads(line)
    text = data['content']
    text = re.sub(" ","",text)
    entities = []
    for annotation in data['annotation']:
        point = annotation['points'][0]
        labels = annotation['label']
        if not isinstance(labels, list):
            labels = [labels]

    for label in labels:
        entities.append((point['start'], point['end'] + 1 ,label))


    training_data.append((text, {"entities" : entities}))

        
# New entity labels
# Specify the new entity labels which you want to add here
LABEL = ['I-geo', 'B-geo', 'I-art', 'B-art', 'B-tim', 'B-nat', 'B-eve', 'O', 'I-per', 'I-tim', 'I-nat', 'I-eve', 'B-per', 'I-org', 'B-gpe', 'B-org', 'I-gpe']

"""
geo = Geographical Entity
org = Organization
per = Person
gpe = Geopolitical Entity
tim = Time indicator
art = Artifact
eve = Event
nat = Natural Phenomenon
"""
# Loading training data 

TRAIN_DATA = training_data
model = en_core_web_sm.load()
new_model_name = 'new_model'
output_dir = "C:/Users/7325598/Desktop/tensorflow practice/1014_4361_bundle_archive/"
n_iter = 10
"""Setting up the pipeline and entity recognizer, and training the new entity."""

nlp = model

if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)
else:
    ner = nlp.get_pipe('ner')

for i in LABEL:
    ner.add_label(i)   # Add new entity labels to entity recognizer

if model is None:
    optimizer = nlp.begin_training()
else:
    optimizer = nlp.entity.create_optimizer()

# Get names of other pipes to disable them during training to train only NER
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.35,losses=losses)
        print('Losses', losses)

# Test the trained model
test_text = 'Gianni Infantino is the president of FIFA.'
doc = nlp(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent.label_, ent.text)



