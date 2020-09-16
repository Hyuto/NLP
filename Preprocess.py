from pandas import DataFrame
from emoji import UNICODE_EMOJI
from string import punctuation
from tqdm.notebook import tqdm
from re import sub

class PuncEmoji(object):
    """
    Untuk mencari sebaran punctuation & emoji di setiap kelasnya
    """
    def __init__(self):
        self.class_ = None  # Daftar Kelas
        self.punc_, self.emo_ = None, None # Df Punctuation & Emoji per kelasnya
        
    @staticmethod
    def is_emoji(word:str) -> bool:
        """
        Mengecek apakah string adalah emoji
        """
        return word in UNICODE_EMOJI
        
    def _punctuation(self, text:str, label):
        """
        Mengecek semua punctuation dalam sebuah kalimat
        """
        for char in text:   # Loop through :v
            if char in punctuation:    # Checking char
                if char not in self.punc_.Punctuation.values:
                    self.punc_.loc[len(self.punc_)] = [char] + [0] * len(self.class_)
                index = self.punc_.Punctuation.tolist().index(char)
                self.punc_.loc[index, label] += 1   # Update Df

    def _emoji(self, text, label):
        """
        Mengecek semua emoji yang terdapat dalam sebuah kalimat
        """
        text = text.split()
        for word in text:   # Loop through :v
            if self.is_emoji(word):   # Checking emoji
                if word not in self.emo_.Emoji.values:
                    self.emo_.loc[len(self.emo_)] = [word] + [0] * len(self.class_)
                index = self.emo_.Emoji.tolist().index(word)
                self.emo_.loc[index, label] += 1    # Update Df
                    
    def fit(self, texts, label):
        """
        Fitting dengan kalimat

        params:
        texts   : List / Numpy array yang berisi kalimat - kalimat yang akan di cek
        label   : List / Numpy array yang berisi kelas dari masing masing kalimat
        """
        self.class_ = list(sorted(set(label)))  # Get Class
        self.punc_ = DataFrame(columns = ['Punctuation'] + self.class_) # Init punctuation Df
        self.emo_ = DataFrame(columns = ['Emoji'] + self.class_) # Init emoji Df
        
        for i, text in enumerate(tqdm(texts)):  # Loop through :v
            self._punctuation(text, label[i])
            self._emoji(text, label[i])
        
        # Sorting Df dari yang terbesar ke yang terkecil 
        self.punc_ = self.punc_.sort_values(by = self.class_, ascending = False).reset_index(drop = True)
        self.emo_ = self.emo_.sort_values(by = self.class_, ascending = False).reset_index(drop = True)

class SpellChecker(object):
    """
    Mengecek dan memperbaiki misspelling words / kata - kata yang typo pada kalimat -
    kalimat yang ada.
    """
    def __init__(self):
        self.words = {}    # Words
        
    def fit(self, direc:str):
        """
        Fitting untuk mendapatkan vocab yang salah dan vocab yang benar dari file txt

        param:
        direc   : Direktori file txt yang berisi vocab
        """
        f = open(direc, "r")
        for w in f.readlines():
            if len(w.split()) == 2:
                missed, true = w.split()
                if '_' in true:
                    true = ' '.join(true.split('_'))
                self.words[missed] = true
        f.close()
        
    def transform(self, arr):
        """
        Mengganti kata - kata yang misspell berdasarkan vocab.

        params:
        arr     : List / Numpy array yang berisi kalimat - kalimat yang akan dibenarkan.
        """
        for i in tqdm(range(len(arr))):
            for miss in self.words:
                arr[i] = sub(miss, self.words[miss] + ' ', arr[i])
                arr[i] = ' '.join(arr[i].split())
        return arr