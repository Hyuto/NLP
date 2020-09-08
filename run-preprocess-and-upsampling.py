# Imports
import os, random, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.logging.set_verbosity(tf.logging.ERROR)

def load_and_preprocess_image(path: str, size = [256, 256]) -> np.ndarray:
    """
    Load & Preprocess Text

    Params
    path           : Image Path
    size           : Image resizing plan [Auto 256 x 256]

    Return
    Numpy ndarray
    """
    # Load image to array and then to tensor
    image = img_to_array(load_img(path))
    img = tf.convert_to_tensor(image, dtype=tf.float32)

    # Resampling image to it's center by square
    shapes = tf.shape(img)
    h, w = shapes[-3], shapes[-2]
    dim = tf.minimum(h, w)
    img = tf.image.resize_with_crop_or_pad(img, dim, dim)

    # Resize
    img = tf.image.resize(img, size)

    # Normalize
    img = tf.cast(img, tf.float32) / 255.0

    return img.numpy() # to Numpy

def data_augmentation(x) -> np.ndarray:
    """
    Image Augmentation. Random rotation in range -70 to 70 degree.

    Params
    X               : Array of image

    Return
    Numpy ndarray
    """
    return rotate(x, random.randint(-70, 70), mode='reflect')

def ApplyAUG(TRAIN_X, TRAIN_y, PATH:str, LP, data_aug, up_sample_ratio = 0.2,
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

    def __up_sampling(up_sample_ratio:float, N:int) -> np.ndarray:
        """
        Up Sampling Plan. Randomly select image data according to up_sample_ratio.
        """
        # Get Div and Mod
        if up_sample_ratio >= 1:
            div, mod = divmod(up_sample_ratio, 1)
        else:
            div, mod = 0, up_sample_ratio
        
        # Sample n file from populations
        n_sample = random.sample(range(N), int(N * mod))
        # Add to main array
        n_AUG = np.array([div] * N) + np.array([1 if i in n_sample else 0 for i in range(N)])

        return np.asarray(n_AUG, dtype = 'int32')

    if up_sample_class != None:
        print(f'[INFO] Up Sampling Kelas {up_sample_class} Sebesar {up_sample_ratio * 100}%')
    else:
        print(f'[INFO] Up Sampling Setiap Kelas Sebesar {up_sample_ratio * 100}%')
    
    TRAIN_X, TRAIN_y = np.asarray(TRAIN_X), np.asarray(TRAIN_y)  # Make sure TRAIN_X & TRAIN_Y is np.array
    os.makedirs(DIR)  # Make Parent Directory
    X, Y = [], []     # Initialize X and Y
    for i in np.unique(TRAIN_y):                    # Loop for every class
        print(f'[INFO] Memproses Kelas {i}..')
        CHILD_DIR = os.path.join(DIR, f'{i}')       # Child/Class Directory
        os.makedirs(CHILD_DIR)
        data = TRAIN_X[TRAIN_y == i]                # Selecting Data based on Class

        if up_sample_class != None:                 # Up Sampling Plan
            if str(i) == up_sample_class:
                n_AUG = __up_sampling(up_sample_ratio, len(data))
            else:
                n_AUG = [0] * len(data)
        else:
            n_AUG = __up_sampling(up_sample_ratio, len(data))

        for k, file in enumerate(tqdm(data)):       # Loop Through
            IMAGE_DIR = os.path.join(CHILD_DIR, f'{file[:-4]}.png') # Image save path
            img = LP(PATH + file)   # Load and Preprocess Image
            tf.keras.preprocessing.image.save_img(IMAGE_DIR, img)   # TF save image
            X.append(IMAGE_DIR); Y.append(i)    # Record path and label to X and Y

            for j in range(n_AUG[k]):   # Loop Through Augmentation Up Sample plan
                AUG_DIR = os.path.join(CHILD_DIR, f'AUG {j + 1}_{file[:-4]}.png') # AUG save path
                aug = data_aug(img)     # Augmented Image
                tf.keras.preprocessing.image.save_img(AUG_DIR, aug) # TF save augmented image
                X.append(AUG_DIR); Y.append(i) # Record path and label to X and Y
        print(f'[INFO] Selesai Memproses Kelas {i}')
        print('[INFO] ' + f'Banyak Data Kelas {i} setelah proses sebanyak {len(os.listdir(CHILD_DIR))} gambar'.title())
    print(f'[INFO] Saved to {DIR}')
    return X, Y

if __name__ == "__main__":
    # System ARGS
    args = sys.argv
    TRAIN_PATH = args[1]
    TEST_SIZE = float(args[2])
    UP_SAMPLES = [float(x) for x in args[3].split('-')]

    print('[INFO] Starting Program to Preprocess & Upsampling Data Gambar..')
    print('[INFO] Config :')
    print(f'      Image Path       : {TRAIN_PATH}')
    print(f'      Split Valid Size : {TEST_SIZE * 100}%')
    print(f"      Rasio Upsample   : {' , '.join([str(x) for x in UP_SAMPLES])}")
    print()

    # X and Y
    data = pd.read_csv('https://raw.githubusercontent.com/Hyuto/NLP/master/TRAIN.CSV')
    TRAIN_X, VAL_X, TRAIN_y, VAL_y = train_test_split(data.X.values, data.y.values, test_size = TEST_SIZE, 
                                                    random_state = SEED, stratify = data.y.values)

    # Up Sample Train Data
    print('[INFO] Memulai Preprocess dan Augmentasi Pada Data Latih')
    for i, up_sample in enumerate(UP_SAMPLES):
        print(f'[INFO] Tahap {i + 1}')
        DIREC = f'Up-Sample-0-by-{int(up_sample * 100)}%'
        TEMP_X, TEMP_Y = ApplyAUG(TRAIN_X, TRAIN_y, TRAIN_PATH, up_sample_ratio = up_sample, 
                                  DIR = DIREC, up_sample_class = '0', data_aug = data_augmentation,
                                  LP = load_and_preprocess_image)
        df = pd.DataFrame({'DIR' : TEMP_X, 'label' : TEMP_Y})
        df.to_csv(DIREC + '/Keterangan.csv', index = False)
        print()

    # Valid Data
    print('[INFO] Memulai Preprocess pada Data Validitas')
    DIREC = 'Validitas'
    TEMP_X, TEMP_Y = ApplyAUG(VAL_X, VAL_y, TRAIN_PATH, up_sample_ratio = 0, 
            DIR = DIREC, data_aug = data_augmentation,
            LP = load_and_preprocess_image)
    df = pd.DataFrame({'DIR' : TEMP_X, 'label' : TEMP_Y})
    df.to_csv(DIREC + '/Keterangan.csv', index = False)
    print()

    print('[INFO] Selesai')
    print('© Catatan Cakrawala 2020')