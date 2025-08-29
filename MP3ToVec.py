import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import numpy as np
import librosa
import pickle
from tqdm import tqdm
import argparse
import random
from keras.layers import TFSMLayer
from pathlib import Path


from tensorflow.keras.models import load_model as modern_load_model
from tensorflow.keras.layers import SeparableConv2D as _SeparableConv2D, LeakyReLU

def SeparableConv2D_compat(*args, **kwargs):
    # Strip out old/unsupported kwargs
    for bad in [
        "groups", "kernel_initializer", "kernel_regularizer", "kernel_constraint",
        "bias_regularizer", "bias_constraint", "activity_regularizer"
    ]:
        kwargs.pop(bad, None)
    return _SeparableConv2D(*args, **kwargs)

def _cosine_proximity(y_true, y_pred):
    y_true = tf.math.l2_normalize(y_true, axis=-1)
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)
    return -tf.reduce_sum(y_true * y_pred, axis=-1)

def _load_h5_or_keras(model_path: str):
    """Try to load model from speccy_model (HDF5)."""
    candidates = [model_path]
    root, ext = os.path.splitext(model_path)
    if not ext:
        candidates += [model_path + ".h5", model_path + ".keras"]

    last_err = None
    for p in candidates:
        if not os.path.isfile(p):
            continue
        try:
            mdl = load_model(
                p,
                custom_objects={
                    "cosine_proximity": _cosine_proximity,
                    "SeparableConv2D": SeparableConv2D_compat,
                    "LeakyReLU": LeakyReLU,  # <-- key fix
                },
                compile=False,
            )
            return mdl, int(mdl.input_shape[1]), int(mdl.input_shape[2])
        except Exception as e:
            last_err = e
            continue
    raise FileNotFoundError(
        f"Could not load model. Tried: {', '.join(candidates)}. Last error: {last_err}"
    )

def walkmp3s(folder):
    for dirpath, dirs, files in os.walk(folder, topdown=False):
        for filename in files:
            if filename.lower().endswith(('.flac', '.mp3', '.m4a')):
                yield filename, os.path.abspath(os.path.join(dirpath, filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pickles', type=str, help='Directory of pickled TrackToVecs')
    parser.add_argument('mp3tovec', type=str, help='Output filename (without extension) of pickled MP3ToVecs')
    parser.add_argument('--scan', type=str, help='Directory of MP3s and M4As to scan')
    parser.add_argument('--model', type=str, help='Load spectrogram to Track2Vec model (default: "speccy_model")')
    parser.add_argument('--batchsize', type=int, help='Number of MP3s to process in each batch (default: 100)')
    parser.add_argument('--epsilon', type=float, help='Epsilon distance (default: 0.001)')
    args = parser.parse_args()
    mp3_directory = args.scan
    dump_directory = args.pickles
    mp3tovec_file = args.mp3tovec
    model_file      = args.model or "speccy_model"
    batch_size = args.batchsize
    epsilon_distance = args.epsilon
    if model_file == None:
        model_file = 'speccy_model'
    if batch_size == None:
        batch_size = 100
    if epsilon_distance == None:
        epsilon_distance = 0.001 # should be small, but not too small
    if not os.path.isdir(dump_directory + '/mp3tovecs'):
        os.makedirs(dump_directory + '/mp3tovecs')
    if mp3_directory is not None:
        print(f'Creating Track2Vec matrices')
        model_path = model_file or 'speccy_model'  # your file is HDF5 even if it has no extension
        mdl, n_mels, slice_size = _load_h5_or_keras(model_path)

        sr         = 22050
        n_fft      = 2048
        hop_length = 512
        slice_time = slice_size * hop_length / sr
        files = []
        done = os.listdir(dump_directory)
        for filename, full_path in walkmp3s(mp3_directory):
            pickle_filename = os.path.splitext(full_path)[0].replace('\\', '_').replace('/', '_').replace(':','_') + '.p'
            if pickle_filename in done:
                continue
            files.append((pickle_filename, full_path))
        random.shuffle(files)
        try:
            with tqdm(files, unit="file") as t:
                for pickle_filename, full_path in t:
                    try:
                        y, sr = librosa.load(full_path, mono=True)
                        if y.shape[0] < slice_size:
                            print(f'Skipping {full_path}')
                            continue
                        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=sr/2)
                        x = np.ndarray(shape=(S.shape[1] // slice_size, n_mels, slice_size, 1), dtype=float)
                        for slice in range(S.shape[1] // slice_size):
                            log_S = librosa.power_to_db(S[:, slice * slice_size : (slice+1) * slice_size], ref=np.max)
                            if np.max(log_S) - np.min(log_S) != 0:
                                log_S = (log_S - np.min(log_S)) / (np.max(log_S) - np.min(log_S))
                            x[slice, :, :, 0] = log_S
                        model_out = mdl.predict(x, verbose=0)  # returns np.ndarray
                        pickle.dump(
							(full_path, model_out),
							open(dump_directory + '/' + pickle_filename, 'wb')
						)
                    except KeyboardInterrupt:
                        raise
                    except:
                        print(f'Skipping {full_path}')
                        continue
        except KeyboardInterrupt:
            t.close() # stop the progress bar from sprawling all over the place after a keyboard interrupt
            raise
        t.close()
    mp3tovecs_fullpath = dump_directory + f'/mp3tovecs/{mp3tovec_file}.p'
    if os.path.isfile(mp3tovecs_fullpath):
        mp3tovecs = pickle.load(open(mp3tovecs_fullpath, 'rb'))
    else:
        mp3tovecs = {}
    unpickled = {}
    for filename in os.listdir(dump_directory):
        if not os.path.isfile(dump_directory + '/' + filename):
            continue
        try:
            p = pickle.load(open(dump_directory + '/' + filename, 'rb'))
            if p[0] in mp3tovecs:
                continue
        except:
            print(f'Skipping pickle {filename}')
            continue
        unpickled[p[0]] = p[1]
    total_num_mp3s = len(unpickled)
    start_batch = 1
    for filename in os.listdir(dump_directory + '/mp3tovecs'):
        if filename[:len(mp3tovec_file)] == mp3tovec_file and filename[len(mp3tovec_file)+1:-2].isdigit():
            mp3tovec = pickle.load(open(dump_directory + '/mp3tovecs/' + filename, 'rb'))
            start_batch += 1
            for mp3 in mp3tovec:
                mp3tovecs[mp3] = mp3tovec[mp3]
    remaining_mp3s = []
    for mp3 in unpickled:
        if mp3 not in mp3tovecs:
            remaining_mp3s.append(mp3)
    num_mp3s = len(remaining_mp3s)
    indices = np.random.permutation(num_mp3s)
    for batch in range(num_mp3s//batch_size + 1):
        print(f'Creating MP3ToVecs for batch {start_batch + batch}/{total_num_mp3s//batch_size + 1}')
        mp3s = {}
        for i in range(batch_size):
            if batch * batch_size + i >= len(indices):
                break
            index = indices[batch * batch_size + i]
            mp3s[remaining_mp3s[index]] = unpickled[remaining_mp3s[index]]
        mp3_vecs = []
        mp3_indices = {}
        for mp3 in mp3s:
            mp3_indices[mp3] = []
            for mp3_vec in mp3s[mp3]:
                mp3_indices[mp3].append(len(mp3_vecs))
                mp3_vecs.append(mp3_vec / np.linalg.norm(mp3_vec)) # normalize
        num_mp3_vecs = len(mp3_vecs)
        # this takes up a lot of memory
        cos_distances = np.zeros((num_mp3_vecs, num_mp3_vecs), dtype=np.float16)
        print(f'Precalculating cosine distances')
        # this needs speeding up
        try:
            with tqdm(mp3_vecs, unit="vector") as t:
                for i, mp3_vec_i in enumerate(t):
                    for j , mp3_vec_j in enumerate(mp3_vecs):
                        if i < j:
                            cos_distances[i, j] = 1 - np.dot(mp3_vec_i, mp3_vec_j)
            cos_distances = cos_distances + cos_distances.T - np.diag(np.diag(cos_distances)) # Make matrix symmetrical diagonally
        except KeyboardInterrupt:
            t.close() # stop the progress bar from sprawling all over the place after a keyboard interrupt
            raise
        t.close()        
        print(f'Calculating IDF weights')
        idfs = []
        try:
            with tqdm(range(num_mp3_vecs), unit="vector") as t:
                for i in t:
                    idf = 0
                    for mp3 in mp3s:
                        for j in mp3_indices[mp3]:
                            if cos_distances[i, j] < epsilon_distance:
                                idf += 1 
                                break
                    idfs.append(-np.log(idf / len(mp3s)))
        except KeyboardInterrupt:
            t.close() # stop the progress bar from sprawling all over the place after a keyboard interrupt
            raise
        t.close()
        print(f'Calculating TF weights')
        mp3tovec = {}
        try:
            with tqdm(mp3s, unit="mp3") as t:
                for mp3 in t:
                    vec = 0
                    for i in mp3_indices[mp3]:
                        tf = 0
                        for j in mp3_indices[mp3]:
                            if cos_distances[i, j] < epsilon_distance:
                                tf += 1
                        vec += mp3_vecs[i] * tf * idfs[i]
                        mp3tovec[mp3] = vec
                        mp3tovecs[mp3] = vec
        except KeyboardInterrupt:
            t.close() # stop the progress bar from sprawling all over the place after a keyboard interrupt
            raise
        t.close()
        pickle.dump(mp3tovec, open(dump_directory + f'/mp3tovecs/{mp3tovec_file}_{start_batch + batch}.p', 'wb'))
        # free up memory
        del cos_distances
        cos_distances = None        
    pickle.dump(mp3tovecs, open(mp3tovecs_fullpath, 'wb'))
    for filename in os.listdir(dump_directory + '/mp3tovecs'):
        if filename[:len(mp3tovec_file)] == mp3tovec_file and filename[len(mp3tovec_file)+1:-2].isdigit():
            os.remove(dump_directory + '/mp3tovecs/' + filename)
