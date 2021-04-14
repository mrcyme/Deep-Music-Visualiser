import librosa
import librosa.display
import numpy as np
import moviepy.editor as mpy
import matplotlib.pyplot as plt
from pydub import AudioSegment
import cv2


def generate_spec_vector(filename, frameLenght, n_fft=2048, n_mels=128):
    y, sr = librosa.load(filename)
    spec = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=frameLenght, n_mels=n_mels)
    specm = np.mean(spec, axis=0)
    gradm = np.gradient(specm)
    gradm = gradm.clip(min=0)
    specm = (specm-np.min(specm))/np.ptp(specm)
    gradm = (gradm-np.min(gradm))/np.ptp(gradm)
    return specm, gradm


def generate_chroma_vector(filename, frameLenght):
    y, sr = librosa.load(filename)
    chroma = librosa.feature.chroma_cqt(y, sr=sr, hop_length=frameLenght)
    chromasort = np.argsort(np.mean(chroma, axis=1))[::-1]
    return chroma, chromasort


def plot_spectogram(spec, sr, frameLenght):
    specDb = librosa.power_to_db(spec, ref=np.max)
    librosa.display.specshow(specDb, sr=sr, hop_length=frameLenght, x_axis='time', y_axis='mel');
    plt.colorbar(format='%+2.0f dB')


def plot_chromogram(chroma):
    librosa.display.specshow(chroma, y_axis='chroma')
    plt.colorbar()
    plt.show()


def make_movie(frames, songfile, outfile, duration=None, frameLenght=512):
    aud = mpy.AudioFileClip(songfile, fps=44100)
    if duration:
        aud.duration = duration
    clip = mpy.ImageSequenceClip(frames, fps=22050/frameLenght)
    clip = clip.set_audio(aud)
    clip.write_videofile(outfile, audio_codec='aac')


def get_update_dir(last_vec, intensity=0.05, truncation=1, tempo_sensitivity=0.25):
    update_dir = np.empty(len(last_vec))
    for ni, n in enumerate(last_vec):
        if n >= 2*truncation - tempo_sensitivity:
            update_dir[ni] = -intensity
        else:
            update_dir[ni] = intensity
    return update_dir


def update_jitters(jitter, vector_length):
    jitters = np.zeros(vector_length)
    for j in range(vector_length):
        if np.random.uniform(0, 1) < 0.5:
            jitters[j] = 1
        else:
            jitters[j] = 1-jitter
    return jitters


def generate_noise_vectors(specm, gradm, starting_vector, truncation=1, tempo_sensitivity=0.25, jitter=0.5):
    total_frames = specm.shape[0]
    vector_length = starting_vector.shape[0]
    vectors = np.zeros((total_frames, vector_length))
    last_vec = starting_vector
    last_update = get_update_dir(last_vec, truncation, tempo_sensitivity)
    for i in range(total_frames):
        if i % 200 == 0:
            jitters = update_jitters(jitter, vector_length)
    update_dir = get_update_dir(last_vec, truncation, tempo_sensitivity)
    update = np.array([tempo_sensitivity for k in range(vector_length)]) * gradm[i] * specm[i] * update_dir * jitters
    update = (update+last_update*3) / 4
    vec = last_vec+update
    vectors[i] = vec
    last_vec = vec
    last_update = update
    return vectors


def generate_class_vector(chroma, chromasort):
    class_vec = np.ones(chroma.T.shape[0]).astype("int")
    for i, c in enumerate(chroma.T):
        class_vec[i] = np.argmax(c)
    return class_vec


def cut_song(songpath, outpath, start, stop):
    song = AudioSegment.from_mp3(songpath)
    extract = song[start*1000:stop*1000]
    extract.export(outpath, format="mp3")


def split_movie(moviepath, time_limits=None):
    frames = []
    upper_limit = 10000000
    vidcap = cv2.VideoCapture(moviepath)
    if time_limits:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, time_limits[0]*1000)
    upper_limit = time_limits[1]*1000
    success, image = vidcap.read()
    count = 0
    while success and vidcap.get(cv2.CAP_PROP_POS_MSEC)<=upper_limit:
        frames.append(image)
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    cv2.destroyAllWindows()
    return frames


def split_movie_and_write(moviepath, framepath, time_limits=None):
    vidcap = cv2.VideoCapture(moviepath)
    upper_limit = 10000000
    if time_limits:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, time_limits[0]*1000)
    upper_limit = time_limits[1]*1000
    success, image = vidcap.read()
    count = 0
    while success and vidcap.get(cv2.CAP_PROP_POS_MSEC) <= upper_limit:
        cv2.imwrite("{}/frame{}.jpg".format(framepath, count), image)
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    cv2.destroyAllWindows()
