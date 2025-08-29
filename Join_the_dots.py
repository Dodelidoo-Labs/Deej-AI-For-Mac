#!/usr/bin/env python3
import os
import pickle
import numpy as np
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.flac import FLAC
import subprocess as sp
import random
import argparse

max_duration = 10 * 60  # avoid adding long mixes to a mix

# ----------------------- durations -----------------------
def get_track_duration(filename):
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == ".mp3":
            return MP3(filename).info.length
        if ext == ".m4a":
            return MP4(filename).info.length
        if ext == ".flac":
            return FLAC(filename).info.length
    except Exception:
        pass
    return 0.0

# ------------------- similarity helpers ------------------
def most_similar(positive=[], negative=[], topn=5, noise=0):
    if isinstance(positive, str):
        positive = [positive]
    if isinstance(negative, str):
        negative = [negative]
    mp3_vec_i = np.sum([mp3tovec[i] for i in positive] + [-mp3tovec[i] for i in negative], axis=0)
    if noise:
        mp3_vec_i += np.random.normal(0, noise * np.linalg.norm(mp3_vec_i), len(mp3_vec_i))
    base_norm = np.linalg.norm(mp3_vec_i)
    similar = []
    for track_j, vj in mp3tovec.items():
        if track_j in positive or track_j in negative:
            continue
        cj = float(np.dot(mp3_vec_i, vj) / (base_norm * norms[track_j]))
        similar.append((track_j, cj))
    return sorted(similar, key=lambda x: -x[1])[:topn]

def most_similar_by_vec(positive=[], negative=[], topn=5, noise=0):
    if isinstance(positive, str):
        positive = [positive]
    if isinstance(negative, str):
        negative = [negative]
    mp3_vec_i = np.sum([i for i in positive] + [-i for i in negative], axis=0)
    if noise:
        mp3_vec_i += np.random.normal(0, noise * np.linalg.norm(mp3_vec_i), len(mp3_vec_i))
    base_norm = np.linalg.norm(mp3_vec_i)
    similar = []
    for track_j, vj in mp3tovec.items():
        cj = float(np.dot(mp3_vec_i, vj) / (base_norm * norms[track_j]))
        similar.append((track_j, cj))
    return sorted(similar, key=lambda x: -x[1])[:topn]

def in_playlist(candidate, playlist):
    if candidate in playlist:
        return True
    vi = mp3tovec[candidate]; ni = norms[candidate]
    for track in playlist:
        vj = mp3tovec[track]; nj = norms[track]
        cos = float(np.dot(vi, vj) / (ni * nj))
        if cos > 1 - epsilon_distance:
            return True
    return False

def make_playlist(seed_tracks, size=10, lookback=3, noise=0):
    max_tries = 100
    playlist = list(seed_tracks)
    while len(playlist) < size:
        similar = most_similar(positive=playlist[-lookback:], topn=max_tries, noise=noise)
        candidates = [c for (c, _) in similar if c != playlist[-1]]
        chosen = None
        for c in candidates:
            if not in_playlist(c, playlist) and get_track_duration(c) < max_duration:
                chosen = c
                break
        if chosen is None:
            # fallback: pick the top unseen
            for c in candidates:
                if c not in playlist:
                    chosen = c
                    break
        if chosen is None:
            # absolute fallback: random
            chosen = random.choice(list(mp3tovec.keys()))
        playlist.append(chosen)
    return playlist

def join_the_dots(tracks, n=5, noise=0):
    max_tries = 100
    playlist = []
    start = tracks[0]
    start_vec = mp3tovec[start]
    for end in tracks[1:]:
        end_vec = mp3tovec[end]
        playlist.append(start)
        for i in range(n):
            alpha = (n - i) / (n + 1)
            target = alpha * start_vec + (1 - alpha) * end_vec
            similar = most_similar_by_vec(positive=[target], topn=max_tries, noise=noise)
            candidates = [c for (c, _) in similar if c != playlist[-1]]
            chosen = None
            for c in candidates:
                if not in_playlist(c, playlist) and c != end and get_track_duration(c) < max_duration:
                    chosen = c
                    break
            if chosen is None:
                for c in candidates:
                    if c != end:
                        chosen = c
                        break
            if chosen is None:
                chosen = random.choice(list(mp3tovec.keys()))
            playlist.append(chosen)
        start = end
        start_vec = end_vec
    playlist.append(tracks[-1])
    return playlist

# --------------------- path handling ---------------------
def relative_path(track, library_root=None, playlist_base=None, fileout=None):
    """
    - If library_root is provided → path relative to that root; optionally prefix playlist_base ('../' etc.).
    - Else → path relative to the playlist file location.
    - If rel escapes root, fall back to absolute.
    """
    track_abs = os.path.realpath(track)

    if library_root:
        root_abs = os.path.realpath(library_root)
        try:
            rel = os.path.relpath(track_abs, start=root_abs)
            if rel.startswith(".."):
                return track_abs
            return os.path.join(playlist_base or "", rel) if playlist_base else rel
        except Exception:
            return track_abs

    # fallback: relative to the playlist file folder
    base = os.path.dirname(os.path.realpath(fileout)) if fileout else os.getcwd()
    return os.path.relpath(track_abs, start=base)

def write_m3u(fileout, tracks, library_root=None, playlist_base=None):
    with open(fileout, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        for t in tracks:
            rel = relative_path(t, library_root=library_root, playlist_base=playlist_base, fileout=fileout)
            f.write(rel + "\n")

# ------------------------- main --------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Join the dots between seed tracks to build a journey; output .m3u or a mixed audio file.")
    ap.add_argument('mp3tovec', type=str, help='Path to mp3tovec pickle (e.g., Pickles/mp3tovecs/mp3tovec.p)')
    ap.add_argument('--inputs', type=str, help='Text file with list of seed songs (one path per line). If omitted, interactive search.')
    ap.add_argument('output', type=str, help='Output filename: use .m3u to write a playlist; any other extension will render audio via ffmpeg')
    ap.add_argument('n', type=int, help='Number of connector songs between each pair of input songs (use 0 to jump directly)')
    ap.add_argument('--noise', type=float, help='Degree of randomness (0–1). Higher = more adventurous.', default=0.0)
    ap.add_argument('--epsilon', type=float, help='Epsilon distance for “already similar” checks (default: 0.001)', default=0.001)
    ap.add_argument('--lookback', type=int, help='(Not used here; only Deej-A.I uses lookback)', default=3)
    # new path-adjustment args
    ap.add_argument('--libraryroot', type=str, help='Root of your music library; m3u entries will be relative to this')
    ap.add_argument('--playlistbase', type=str, help="Optional prefix to prepend to each m3u entry (e.g. '../')")

    args = ap.parse_args()
    mp3tovec_path = args.mp3tovec
    output = args.output
    n = args.n
    noise = float(args.noise or 0.0)
    epsilon_distance = float(args.epsilon or 0.001)

    # load vectors
    mp3tovec = pickle.load(open(mp3tovec_path, 'rb'))
    norms = {k: float(np.linalg.norm(v)) for k, v in mp3tovec.items()}

    # seeds
    input_tracks = []
    if args.inputs:
        with open(args.inputs, 'rt', encoding='utf-8') as fh:
            for line in fh:
                t = line.strip()
                if t:
                    input_tracks.append(t)
    else:
        # minimal interactive search (unchanged behavior)
        user_input = input('Search keywords: ')
        while True:
            if user_input == '':
                break
            tracks = sorted([mp3 for mp3 in mp3tovec if all(word in mp3.lower() for word in user_input.lower().split())])
            for i, track in enumerate(tracks):
                print(f'{i+1}. {track}')
            while True:
                user_input = input('Input track number to add, ENTER to finish, or search keywords: ')
                if user_input == '':
                    break
                if user_input.isdigit() and len(tracks) > 0:
                    idx = int(user_input) - 1
                    if 0 <= idx < len(tracks):
                        input_tracks.append(tracks[idx])
                        print(f'Added {tracks[idx]}')
                else:
                    break
        print()

    if not input_tracks:
        # if nothing provided, pick a random seed
        all_tracks = list(mp3tovec.keys())
        input_tracks = [random.choice(all_tracks)]

    # build playlist
    if len(input_tracks) > 1:
        playlist = join_the_dots(input_tracks, n=n, noise=noise)
    else:
        # with one seed, make a straight list of size n (not n connectors)
        playlist = make_playlist(input_tracks, size=n or 10, lookback=3, noise=noise)

    # pretty print & duration
    total = 0.0
    for i, t in enumerate(playlist, 1):
        total += get_track_duration(t)
        star = "*" if (n == 0 and i == 1) or (n != 0 and (i - 1) % (n + 1) == 0) else ""
        print(f"{i}.{star} {t}")
    print(f"Total duration = {int(total)//3600:d}:{(int(total)//60)%60:02d}:{int(total)%60:02d}s\n")

    # write output
    if output.lower().endswith(".m3u"):
        write_m3u(output, playlist, library_root=args.libraryroot, playlist_base=args.playlistbase)
        print(f"Wrote playlist: {output}")
    else:
        # original mix renderer via ffmpeg
        print(f"Creating mix {output}")
        tracks_args = []
        for t in playlist:
            tracks_args += ['-i', t]
        try:
            pipe = sp.Popen(
                ['ffmpeg', '-y', '-i', 'static/meta_data.txt'] +
                tracks_args +
                ['-filter_complex', f'loudnorm=I=-14,concat=n={len(playlist)}:v=0:a=1[out]',
                 '-map', '[out]', output],
                stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE
            )
            out, err = pipe.communicate()
            if pipe.returncode != 0:
                print("ffmpeg failed. stderr:\n", err.decode(errors='ignore'))
        except FileNotFoundError:
            print("ffmpeg is not installed. To write a playlist instead, use a .m3u output filename.")
