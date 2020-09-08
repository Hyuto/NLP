import os, random, sys
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from skimage.transform import rotate

# SEED ALL
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

def load_and_preprocess_image(path: str, size = [256, 256]):
    image = img_to_array(load_img(path))
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    shapes = tf.shape(img)
    h, w = shapes[-3], shapes[-2]
    dim = tf.minimum(h, w)
    img = tf.image.resize_with_crop_or_pad(img, dim, dim)
    img = tf.image.resize(img, size)
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()

def data_augmentation(x):
    return rotate(x, random.randint(-70, 70), mode='reflect')

def ApplyAUG(TRAIN_X, TRAIN_y, PATH, LP, data_aug, up_sample_ratio = 0.2,
             up_sample_class = None, DIR = 'Prep Data + AUG'):
    """
    Fungsi untuk mengaplikasikan Preprocess dan Augmentation ke dalam data gambar untuk disimpan
    kedalam direktori/file yang baru.

    Params
    TRAIN_X         : List atau Array direktori dari gambar
    TRAIN_y         : List atau Array kelas(label) dari TRAIN_X
    PATH            : Direktori data gambar
    LP              : Load & Preprocess image function
    data_aug        : Augmentation function
    up_sample_ratio : Rasio up sampling yang dikehendaki.
    up_sample_class : Spesifikasi kelas yang akan dilakukan Augmentasi. Jika ini di isi maka 
                      upsample hanya akan dilakukan pada kelas yang di spesifikasikan.

    Return
    List item pada direktori baru dan Labelnya.
    """

    def __up_sampling(up_sample_ratio, N):
        if up_sample_ratio >= 1:
            div, mod = divmod(up_sample_ratio, 1)
        else:
            div, mod = 0, up_sample_ratio
        n_sample = random.sample(range(N), int(N * mod))
        n_AUG = np.array([div] * N) + np.array([1 if i in n_sample else 0 for i in range(N)])
        return np.asarray(n_AUG, dtype = 'int32')

    if up_sample_class != None:
        print(f'[INFO] Up Sampling Kelas {up_sample_class} Sebesar {up_sample_ratio * 100}%')
    else:
        print(f'[INFO] Up Sampling Setiap Kelas Sebesar {up_sample_ratio * 100}%')
    
    TRAIN_X, TRAIN_y = np.asarray(TRAIN_X), np.asarray(TRAIN_y)
    os.makedirs(DIR)
    X, Y = [], []
    for i in np.unique(TRAIN_y):
        print(f'[INFO] Memproses Kelas {i}..')
        CHILD_DIR = os.path.join(DIR, f'{i}')
        os.makedirs(CHILD_DIR)
        data = TRAIN_X[TRAIN_y == i]
        if up_sample_class != None:
            if str(i) == up_sample_class:
                n_AUG = __up_sampling(up_sample_ratio, len(data))
            else:
                n_AUG = [0] * len(data)
        else:
            n_AUG = __up_sampling(up_sample_ratio, len(data))
        for k, file in enumerate(tqdm(data)):
            IMAGE_DIR = os.path.join(CHILD_DIR, f'{file[:-4]}.png')
            img = LP(PATH + file)
            tf.keras.preprocessing.image.save_img(IMAGE_DIR, img)
            X.append(IMAGE_DIR); Y.append(i)
            for j in range(n_AUG[k]):
                AUG_DIR = os.path.join(CHILD_DIR, f'AUG {j + 1}_{file[:-4]}.png')
                aug = data_aug(img)
                tf.keras.preprocessing.image.save_img(AUG_DIR, aug)
                X.append(AUG_DIR); Y.append(i)
        print(f'[INFO] Selesai Memproses Kelas {i}')
        print('[INFO] ' + f'Banyak Data Kelas {i} setelah proses sebanyak {len(os.listdir(CHILD_DIR))} gambar\n'.title())
    print(f'[INFO] Saved to {DIR}')
    print('[INFO] Done :)')
    return X, Y

if __name__ == "__main__":
    args = sys.argv
    TRAIN_PATH = args[1]
    TEST_SIZE = float(args[2])
    UP_SAMPLES = [float(x) for x in args[3].split('-')]

    data = pd.read_csv('https://raw.githubusercontent.com/Hyuto/NLP/master/TRAIN.CSV')
    TRAIN_X, VAL_X, TRAIN_y, VAL_y = train_test_split(data.X.values, data.y.values, test_size = TEST_SIZE, 
                                                    random_state = SEED, stratify = TRAIN_y)

    for up_sample in UP_SAMPLES:
        DIREC = f'Up-Sample-0-by-{int(up_sample * 100)}%'
        TEMP_X, TEMP_Y = ApplyAUG(TRAIN_X, TRAIN_y, TRAIN_PATH, up_sample_ratio = up_sample, 
                DIR = DIREC, up_sample_class = '0', data_aug = data_augmentation,
                LP = load_and_preprocess_image)
        df = pd.DataFrame({'DIR' : TEMP_X, 'label' : TEMP_Y})
        df.to_csv(DIREC + '/Keterangan.csv', index = False)

    DIREC = 'Validitas'
    TEMP_X, TEMP_Y = ApplyAUG(VAL_X, VAL_y, TRAIN_PATH, up_sample_ratio = 0, 
            DIR = DIREC, data_aug = data_augmentation,
            LP = load_and_preprocess_image)
    df = pd.DataFrame({'DIR' : TEMP_X, 'label' : TEMP_Y})
    df.to_csv(DIREC + '/Keterangan.csv', index = False)