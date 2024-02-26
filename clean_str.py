# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:54:30 2020

@author: ZEEV
"""
import re
def clean_str(text):
    search = [ '_', '-', '/', '.', '،', ',', 'ـ', '\\', '\n', '\t',  '&quot;', '?', '!',':',';']
    # replace = ['ا', 'ا', 'ا', 'ه', ' ', ' ', '', '', '', ' و', ' يا', '', '', '', 'ي', '', ' ', ' ', ' ', ' ? ', ' ؟ ',
               # ' ! ']

    # remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel, '', text)
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r'\1\1'
    text = re.sub(p_longation, subst, text)

    

    for i in range(0, len(search)):
        text = text.replace(search[i],' ')
        
    for i in range(0, len(punctuations)):
        text = text.replace(punctuations[i],' ')
    # trim
    text = text.strip()

    return text