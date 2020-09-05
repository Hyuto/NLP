import os
import tensorflow as tf
import numpy as np
from tqdm.notebook import tqdm

def ApplyAUG(df, PATH, n_AUG = 1, up_sample_class = None, up_sample_ratio = 0.2,
            load_and_preprocess_image = load_and_preprocess_image, data_augmentation = data_augmentation):
    """
    Fungsi untuk mengaplikasikan Preprocess dan Augmentation ke dalam data gambar untuk disimpan
    kedalam direktori/file yang baru.

    Params
    df              : Dataframe yang menyimpan nama file pada kolom 'nama file gambar' dan label pada kolom
                     'label'
    PATH            : Direktori data gambar
    n_AUG           : Banyaknya jumlah Augmentasi yang akan dilakukan kepada setiap data gambar(Otomatis = 1)
    up_sample_class : Spesifikasi kelas yang akan dilakukan Augmentasi. Jika ini di isi maka n_AUG
                      akan dihitung secara otomatis tergantung up_sample_ratio.
    up_sample_ratio : Rasio up sampling yang dikehendaki.
    """
    if up_sample_class != None:
        print(f'[INFO] Up Sampling Kelas {up_sample_class} sebesar {up_sample_ratio * 100}%')
        print(f'[INFO] n_AUG akan dikalkulasi secara otomatis..\n')
    
    DIR = './Prep Data + AUG'
    os.makedirs(DIR)
    for i in range(2):
        print(f'[INFO] Memproses Kelas {i}..')
        CHILD_DIR = os.path.join(DIR, f'{i}')
        os.makedirs(CHILD_DIR)
        data = df['nama file gambar'][df.label.values == i].values
        if up_sample_class != None:
            if str(i) == up_sample_class:
                if up_sample_ratio >= 1:
                    n_AUG = [up_sample_ratio]*len(data)
                else:
                    n_sample = int(len(data) * up_sample_ratio)
                    n_AUG = [1] * n_sample + [0] * (len(data) - n_sample)
            else:
                n_AUG = [0] * len(data)
        else:
            n_AUG = [n_AUG]*len(data)
        for k, file in enumerate(tqdm(data)):
            IMAGE_DIR = os.path.join(CHILD_DIR, f'{file[:-4]}.png')
            img = load_and_preprocess_image(PATH + file)
            tf.keras.preprocessing.image.save_img(IMAGE_DIR, img)
            for j in range(n_AUG[k]):
                AUG_DIR = os.path.join(CHILD_DIR, f'AUG {j + 1}_{file[:-4]}.png')
                aug = data_augmentation(np.expand_dims(img, 0))
                tf.keras.preprocessing.image.save_img(AUG_DIR, aug[0])
        print(f'[INFO] Selesai Memproses Kelas {i}')
        print('[INFO] ' + f'Banyak Data Kelas {i} setelah proses sebanyak {len(os.listdir(CHILD_DIR))} gambar\n'.title())
    print('[INFO] Done :)')