import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import dash
from dash.dependencies import Input, Output, State
import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
from flask import send_from_directory
from io import BytesIO
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.flac import FLAC, Picture
from mutagen.id3 import ID3, ID3NoHeaderError
from PIL import Image
import base64
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
import librosa
import pickle
import random
import shutil
import time
import argparse
from pathlib import PurePosixPath

# use TF Keras everywhere
from tensorflow.keras.models import load_model as tf_load_model
from tensorflow.keras.layers import SeparableConv2D as _SeparableConv2D, LeakyReLU
import tensorflow as tf  # already imported later, but we need it here

# --- legacy loss shim (old 'cosine_proximity') ---
def _cosine_proximity(y_true, y_pred):
    y_true = tf.math.l2_normalize(y_true, axis=-1)
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)
    return -tf.reduce_sum(y_true * y_pred, axis=-1)

# --- SeparableConv2D config shim (strip legacy kwargs that Keras 3 rejects) ---
def SeparableConv2D_compat(*args, **kwargs):
    for bad in [
        "groups", "kernel_initializer", "kernel_regularizer", "kernel_constraint",
        "bias_regularizer", "bias_constraint", "activity_regularizer"
    ]:
        kwargs.pop(bad, None)
    return _SeparableConv2D(*args, **kwargs)

# --- helper to load 'speccy_model' with/without extension ---
def _load_speccy_model(path_like: str):
    candidates = [path_like]
    root, ext = os.path.splitext(path_like)
    if not ext:
        candidates += [path_like + ".h5", path_like + ".keras"]
    last_err = None
    for p in candidates:
        if not os.path.isfile(p):
            continue
        try:
            mdl = tf_load_model(
                p,
                custom_objects={
                    "cosine_proximity": _cosine_proximity,
                    "SeparableConv2D": SeparableConv2D_compat,
                    "LeakyReLU": LeakyReLU,
                },
                compile=False,
            )
            return mdl
        except Exception as e:
            last_err = e
    raise FileNotFoundError(f"Could not load model from {candidates}. Last error: {last_err}")

default_lookback = 3 # number of previous tracks to take into account
default_noise = 0    # amount of randomness to throw in the mix
default_playlist_size = 10

app = dash.Dash()
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

@app.server.route('/static/<path:path>')
def static_file(path):
    static_folder = os.path.join(os.getcwd(), 'static')
    return send_from_directory(static_folder, path)

theme = {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E'
}

upload = html.Div(
    [
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File'),
                ' and wait a bit...'
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center'
            },
            # Do not allow multiple files to be uploaded
            multiple=False
        ),
        html.Br()
    ],
    style={
        'width': '98vw',
        'textAlign': 'center',
        'margin': 'auto'
    }
)

shared = html.Div(
    id='shared-info',
    style={
        'display': 'none'
    }
)

lookback_knob = html.Div(
    [
        daq.Knob(
            id='lookback-knob',
            label='Keep on',
            size=90,
            max=10,
            value=default_lookback,
            scale={
                'start': 0,
                'labelInterval': 10,
                'interval': 1
            },
            theme=theme
        ),
        html.Div(
            id='lookback-knob-output',
            style={
                'display': 'none'
            }
        )
    ],
    style={
        'width': '10%',
        'display': 'inline-block'
    }
)

noise_knob = html.Div(
    [
        daq.Knob(
            id='noise-knob',
            label='Drunk',
            size=90,
            min=0,
            max=1,
            value=default_noise,
            scale={
                'start': 0,
                'labelInterval': 10,
                'interval': 0.1
            },
            theme=theme
        ),
        html.Div(
            id='noise-knob-output',
            style={
                'display': 'none'
            }
        )
    ],
    style={
        'width': '10%',
        'display': 'inline-block'
    }
)

app.layout = html.Div(
    [
        html.Link(
            rel='stylesheet',
            href='/static/custom.css'
        ),
        html.Div(
            [
                html.Div(
                    id='output-image-upload',
                    children=[
                        upload,
                        shared
                    ]
                )
            ]
        ),
        html.Div(
            [
                lookback_knob,
                html.Div(
                    style={
                        'width': '79%',
                        'display': 'inline-block'
                    }
                ),
                noise_knob
            ]
        )
    ]
)

def most_similar(positive=[], negative=[], topn=5, noise=0):
    if isinstance(positive, str):
        positive = [positive] # broadcast to list
    if isinstance(negative, str):
        negative = [negative] # broadcast to list
    mp3_vec_i = np.sum([mp3tovec[i] for i in positive] + [-mp3tovec[i] for i in negative], axis=0)
    mp3_vec_i += np.random.normal(0, noise * np.linalg.norm(mp3_vec_i), len(mp3_vec_i))
    similar = []
    for track_j in mp3tovec:
        if track_j in positive or track_j in negative:
            continue
        mp3_vec_j = mp3tovec[track_j]
        cos_proximity = np.dot(mp3_vec_i, mp3_vec_j) / (np.linalg.norm(mp3_vec_i) * np.linalg.norm(mp3_vec_j))
        similar.append((track_j, cos_proximity))
    return sorted(similar, key=lambda x:-x[1])[:topn]

def most_similar_by_vec(positive=[], negative=[], topn=5, noise=0):
    if isinstance(positive, str):
        positive = [positive] # broadcast to list
    if isinstance(negative, str):
        negative = [negative] # broadcast to list
    mp3_vec_i = np.sum([i for i in positive] + [-i for i in negative], axis=0)
    mp3_vec_i += np.random.normal(0, noise * np.linalg.norm(mp3_vec_i), len(mp3_vec_i))
    similar = []
    for track_j in mp3tovec:
        mp3_vec_j = mp3tovec[track_j]
        cos_proximity = np.dot(mp3_vec_i, mp3_vec_j) / (np.linalg.norm(mp3_vec_i) * np.linalg.norm(mp3_vec_j))
        similar.append((track_j, cos_proximity))
    return sorted(similar, key=lambda x:-x[1])[:topn]

def make_playlist(seed_tracks, size=10, lookback=3, noise=0):
    max_tries = 10
    playlist = seed_tracks
    while len(playlist) < size:
        similar = most_similar(positive=playlist[-lookback:], topn=max_tries, noise=noise)
        candidates = [candidate[0] for candidate in similar if candidate[0] != playlist[-1]]
        for candidate in candidates:
            if not candidate in playlist:
                break
        playlist.append(candidate)
    return playlist

def get_mp3tovec(content_string, filename):
    sr                = 22050
    n_fft             = 2048
    hop_length        = 512
    n_mels            = 96 # model.layers[0].input_shape[1]
    slice_size        = 216 # model.layers[0].input_shape[2]
    slice_time        = slice_size * hop_length / sr
    start = time.time()
    print(f'Analyzing {filename}')
    decoded = base64.b64decode(content_string)
    with open('dummy.' + filename[-3:], 'wb') as file: # this is really annoying!
        shutil.copyfileobj(BytesIO(decoded), file, length=131072)
    y, sr = librosa.load('dummy.' + filename[-3:], mono=True)
    os.remove('dummy.' + filename[-3:])
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=sr/2)
    x = np.ndarray(shape=(S.shape[1] // slice_size, n_mels, slice_size, 1), dtype=float)
    for slice in range(S.shape[1] // slice_size):
        log_S = librosa.power_to_db(S[:, slice * slice_size : (slice+1) * slice_size], ref=np.max)
        if np.max(log_S) - np.min(log_S) != 0:
            log_S = (log_S - np.min(log_S)) / (np.max(log_S) - np.min(log_S))
        x[slice, :, :, 0] = log_S
    # need to put semaphore around this
    K.clear_session()
    model = _load_speccy_model(model_file or "speccy_model")
    new_vecs = model.predict(x, verbose=0)
    K.clear_session()
    print(f'Spectrogram analysis took {time.time() - start:0.0f}s')
    start = time.time()
    try:
        mp3s = {}
        dropout = batch_size / len(mp3tovec) # only process a selection of MP3s
        for vec in mp3tovec:
            if random.uniform(0, 1) > dropout:
                continue
            pickle_filename = (vec[:-3]).replace('\\', '_').replace('/', '_').replace(':','_') + 'p'
            try:
                unpickled = pickle.load(open(dump_directory + '/' + pickle_filename, 'rb'))
            except:
                pickle_filename = pickle_filename.encode('ISO8859-1', 'replace').decode('ascii', 'surrogateescape')
                unpickled = pickle.load(open(dump_directory + '/' + pickle_filename, 'rb'))
            mp3s[unpickled[0]] = unpickled[1]
        new_idfs = []
        for vec_i in new_vecs:
            idf = 1 # because each new vector is in the new mp3 by definition
            for mp3 in mp3s:
                for vec_j in mp3s[mp3]:
                    if 1 - np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j)) < epsilon_distance:
                        idf += 1
                        break
            new_idfs.append(-np.log(idf / (len(mp3s) + 1))) # N + 1
        vec = 0
        for i, vec_i in enumerate(new_vecs):
            tf_ = 0
            for vec_j in new_vecs:
                if 1 - np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j)) < epsilon_distance:
                    tf_ += 1
            vec += vec_i * tf_ * new_idfs[i]
    except:
        vec = np.mean(new_vecs, axis=0)
    similar = most_similar_by_vec([vec], topn=1, noise=0)
    print(f'TF-IDF analysis took {time.time() - start:0.0f}s')
    return similar[0][0]

def get_track_info(filename):
    artwork = pict = None
    artist = track = album = None
    duration = 0
    if filename[-3:].lower() == 'mp3':
        try:
            audio = ID3(filename)
            if audio.get('APIC:') is not None:
                pict = audio.get('APIC:').data
            if audio.get('APIC:Cover') is not None:
                pict = audio.get('APIC:Cover').data
            if pict is not None:
                im = Image.open(BytesIO(pict)).convert('RGB')
                buff = BytesIO()
                im.save(buff, format='jpeg')
                artwork = base64.b64encode(buff.getvalue()).decode('utf-8')
            if audio.get('TPE1') is not None:
                artist = audio['TPE1'].text[0]
            if audio.get('TIT2') is not None:
                track = audio['TIT2'].text[0]
            if audio.get('TALB') is not None:
                album = audio['TALB'].text[0]
        except ID3NoHeaderError:
            pass
        duration = MP3(filename).info.length
    elif filename[-3:].lower() == 'm4a':
        audio = MP4(filename)
        if audio.get("covr") is not None:
            pict = audio.get("covr")[0]
            im = Image.open(BytesIO(pict)).convert('RGB')
            buff = BytesIO()
            im.save(buff, format='jpeg')
            artwork = base64.b64encode(buff.getvalue()).decode('utf-8')
        if audio.get('\xa9ART') is not None:
            artist = audio.get('\xa9ART')[0]
        if audio.get('\xa9nam') is not None:
            track = audio.get('\xa9nam')[0]
        if audio.get('\xa9alb') is not None:
            album = audio.get('\xa9alb')[0]
        duration = audio.info.length
    elif filename[-4:].lower() == 'flac':
        audio = FLAC(filename)
        if audio.pictures:
            for pict in audio.pictures:
                if pict.type == 3:
                    im = Image.open(BytesIO(pict.data)).convert('RGB')
                    buff = BytesIO()
                    im.save(buff, format='jpeg')
                    artwork = base64.b64encode(buff.getvalue()).decode('utf-8')
        if audio.get('ARTIST') is not None:
            artist = audio.get('ARTIST')[0]
        if audio.get('TITLE') is not None:
            track = audio.get('TITLE')[0]
        if audio.get('ALBUM') is not None:
            album = audio.get('ALBUM')[0]
        duration = audio.info.length
    if artwork == None:
        artwork = base64.b64encode(open('./static/record.jpg', 'rb').read()).decode()
    if (artist, track, album) == (None, None, None):
        artist = filename
    return artwork, artist, track, album, duration

def play_track(tracks, durations):
    artwork, artist, track, album, duration = get_track_info(tracks[-1])
    print(f'{len(tracks)}. {artist} - {track} ({album})')
    df = pd.DataFrame({'tracks': tracks, 'durations': durations + [duration]})
    jsonifed_data = df.to_json()
    return html.Div(
        [
            html.H1(
                f'{len(tracks)}. {artist} - {track} ({album})'
            ),
            html.Div(
                dcc.Upload(
                    id='upload-image',
                    style={
                            'display': 'none'
                    }
                )
            ),
            html.Audio(
                id='audio',
                src='data:audio/mp3;base64,{}'.format(base64.b64encode(open(tracks[-1], 'rb').read()).decode()),
                controls=False,
                autoPlay=True,
                style={
                    'display': 'none'
                }
            ),
            html.Div(
                [
                    html.Div(
                        html.Img(
                            src='data:image/jpeg;base64,{}'.format(artwork),
                            style={
                                'width': '85vh',
                                'margin': 'auto',
                                'display': 'inline-block'
                            }
                        ),
                        style={
                            'textAlign': 'center',
                        }
                    )
                ]
            ),
            html.Div(
                id='shared-info',
                style={
                    'display': 'none'
                },
                children=jsonifed_data
            )
        ]
    )

@app.callback(
    Output('lookback-knob-output', 'children'),
    [Input('lookback-knob', 'value')])
def update_output(value):
    print(f'lookback changed to {value}')
    return int(value)

@app.callback(
    Output('noise-knob-output', 'children'),
    [Input('noise-knob', 'value')])
def update_output(value):
    print(f'noise changed to {value}')
    return value

@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents'),
               Input('shared-info', 'children')],
              [State('upload-image', 'filename'),
               State('lookback-knob-output', 'children'),
               State('noise-knob-output', 'children')])
def update_output(contents, jsonified_data, filename, lookback, noise):
    print(f'lookback={lookback}, noise={noise}')
    if lookback is None:
        lookback = default_lookback
    if noise is None:
        noise = default_noise
    if jsonified_data is not None:
        # next time round
        df = pd.read_json(jsonified_data)
        durations = df['durations'].tolist()
        if demo is not None:
            time.sleep(demo)
        else:
            time.sleep(durations[-1])
        tracks = df['tracks'].tolist()
        tracks = make_playlist(tracks, size=len(tracks)+1, noise=noise, lookback=lookback)
        return play_track(tracks, durations)
    if contents is not None and filename is not None:
        # first time round
        content_type, content_string = contents.split(',')
        track = get_mp3tovec(content_string, filename)
        return play_track([track], [])
    # make sure we get called back
    time.sleep(1)
    return [upload, shared]

def relative_path(track, library_root=None, fileout=None):
    """
    Return a path suitable for M3U:
    - If library_root is provided, make the path relative to that root.
    - If the track is not under library_root, fall back to absolute path.
    - Otherwise (no library_root), make it relative to the playlist file location.
    """
    track_abs = os.path.realpath(track)

    if library_root:
        root_abs = os.path.realpath(library_root)
        try:
            rel = os.path.relpath(track_abs, start=root_abs)
            # If it escapes the root (starts with '..'), fallback to absolute
            if rel.startswith('..'):
                return track_abs
            return rel
        except Exception:
            return track_abs

    # Fallback: relative to the playlist file’s folder
    if fileout and os.path.isdir(fileout):
        base = fileout
    else:
        base = os.path.dirname(fileout) if fileout else os.getcwd()
    return os.path.relpath(track_abs, start=os.path.realpath(base))

def _lib_relative(track, library_root):
    # path inside library_root, POSIX separators for M3U
    track_abs = os.path.realpath(track)
    root_abs  = os.path.realpath(library_root) if library_root else None
    if root_abs:
        rel = os.path.relpath(track_abs, start=root_abs)
    else:
        # fallback: just the filename
        rel = os.path.basename(track_abs)
    return rel.replace(os.sep, "/")

def _m3u_entry(track, library_root=None, playlist_base=None):
    rel = _lib_relative(track, library_root)
    if playlist_base is None or playlist_base == "":
        return rel
    # Join using POSIX semantics so "../" etc. are preserved cleanly
    return str(PurePosixPath(playlist_base) / PurePosixPath(rel))

def tracks_to_m3u(fileout, tracks, library_root=None, playlist_base=None):
    with open(fileout, 'w', encoding='utf-8') as f:
        f.write("#EXTM3U\n")
        for item in tracks:
            f.write(_m3u_entry(item, library_root=library_root,
                               playlist_base=playlist_base) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pickles', type=str, help='Directory of pickled TrackToVecs')
    parser.add_argument('mp3tovec', type=str, help='Filename (without extension) of pickled MP3ToVecs')
    parser.add_argument('--demo', type=int, help='Only play this number of seconds of each track')
    parser.add_argument('--model', type=str, help='Load spectrogram to Track2Vec model (default: "speccy_model")')
    parser.add_argument('--batchsize', type=int, help='Number of MP3s to process in each batch (default: 100)')
    parser.add_argument('--epsilon', type=float, help='Epsilon distance (default: 0.001)')
    parser.add_argument('--playlist', type=str, help='Write playlist file without starting interface')
    parser.add_argument('--inputsong', type=str, help="Requires --playlist option\nSelects a song to start the playlist with.")
    parser.add_argument("--nsongs", type=int, help="Requires --playlist option\nNumber of songs in the playlist")
    parser.add_argument("--noise", type=float, help="Requires --playlist option\nAmount of noise in the playlist (default 0)")
    parser.add_argument("--lookback", type=int, help="Requires --playlist option\nAmount of lookback in the playlist (default 3)")
    parser.add_argument('--libraryroot', type=str, help='Path to Navidrome music root (make M3U entries relative to this)')
    parser.add_argument('--playlistbase', type=str,
    help='Prefix to use for each M3U entry (e.g. "../" or absolute base).')

    args = parser.parse_args()
    dump_directory = args.pickles
    mp3tovec_file = args.mp3tovec
    demo = args.demo
    model_file = args.model
    batch_size = args.batchsize
    epsilon_distance = args.epsilon
    playlist_outfile = args.playlist
    input_song = args.inputsong
    n_songs = args.nsongs
    noise = args.noise
    lookback = args.lookback
    library_root = args.libraryroot
    playlist_base = args.playlistbase
    

    if model_file == None:
        model_file = 'speccy_model'
    if batch_size == None:
        batch_size = 100
    if epsilon_distance == None:
        epsilon_distance = 0.001 # should be small, but not too small
    mp3tovec = pickle.load(open(dump_directory + '/mp3tovecs/' + mp3tovec_file + '.p', 'rb'))
    print(f'{len(mp3tovec)} MP3s')
    if playlist_outfile == None:
        app.run_server(threaded=False, debug=False)
    else:
        if input_song != None:
            if n_songs == None:
                n_songs = default_playlist_size
            if noise == None:
                noise = default_noise
            if lookback == None:
                lookback = default_lookback

            print("Outfile playlist: {}".format(playlist_outfile))
            print("Input song selected: {}".format(input_song))
            print("Requested {} songs".format(n_songs))

            tracks = make_playlist([input_song], size=n_songs + 1, noise=noise, lookback=lookback)
            tracks_to_m3u(playlist_outfile, tracks,
              library_root=library_root,
              playlist_base=playlist_base)
        else:
            print("[ERR] Argument --inputsong is required")
