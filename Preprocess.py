from pandas import DataFrame
from emoji import UNICODE_EMOJI
from string import punctuation
from tqdm.notebook import tqdm
from re import sub

class PuncEmoji(object):
    def __init__(self):
        self.class_ = None
        self.punc_, self.emo_ = None, None
        
    @staticmethod
    def is_emoji(word):
        return word in UNICODE_EMOJI
        
    def _punctuation(self, text, label):
        for char in text:
            if char in punctuation:
                if char not in self.punc_.Punctuation.values:
                    self.punc_.loc[len(self.punc_)] = [char] + [0] * len(self.class_)
                index = self.punc_.Punctuation.tolist().index(char)
                self.punc_.loc[index, label] += 1

    def _emoji(self, text, label):
        text = text.split()
        for word in text:
            if self.is_emoji(word):
                if word not in self.emo_.Emoji.values:
                    self.emo_.loc[len(self.emo_)] = [word] + [0] * len(self.class_)
                index = self.emo_.Emoji.tolist().index(word)
                self.emo_.loc[index, label] += 1
                    
    def fit(self, texts, label):
        self.class_ = list(sorted(set(label)))
        self.punc_ = DataFrame(columns = ['Punctuation'] + self.class_)
        self.emo_ = DataFrame(columns = ['Emoji'] + self.class_)
        
        for i, text in enumerate(tqdm(texts)):
            self._punctuation(text, label[i])
            self._emoji(text, label[i])
            
        self.punc_ = self.punc_.sort_values(by = self.class_, ascending = False).reset_index(drop = True)
        self.emo_ = self.emo_.sort_values(by = self.class_, ascending = False).reset_index(drop = True)

class SpellChecker(object):
    def __init__(self):
        self.words = {}
        
    def fit(self, direc):
        f = open(direc, "r")
        for w in f.readlines():
            if len(w.split()) == 2:
                missed, true = w.split()
                if '_' in true:
                    true = ' '.join(true.split('_'))
                self.words[missed] = true
        f.close()
        
    def transform(self, arr):
        for i in tqdm(range(len(arr))):
            for miss in self.words:
                arr[i] = sub(miss, self.words[miss], arr[i])
        return arr