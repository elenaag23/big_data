import librosa
import os
import numpy as np
import pandas as pd
from librosa.feature.rhythm import tempo as rhythm_tempo

base_path = "data/Data/genres_original"
genres = os.listdir(base_path)
features = []

for genre in genres:
    genre_path = os.path.join(base_path, genre)
    for file in os.listdir(genre_path):
        if file.endswith(".wav"):
            path = os.path.join(genre_path, file)
            try:
                y, sr = librosa.load(path)
                tempo_val = rhythm_tempo(y=y, sr=sr)[0] # variatia de ritm a piesei
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean() # inaltimea sunetului
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean() # frecventa valorii 0 in evolutia semnalului - ajuta la identificarea zgomotelor percutante
                rmse = librosa.feature.rms(y=y).mean() # Root Mean Square Energy - energia generala a semnalului
                bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean() # latimea benzii de frecventa - urmareste cat de mult variaza semnalul
                chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean() # distributia semnalului pe notele muzicale - reflecta tonalitatea
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) # Mel-Frequency Cepstral Coefficients - reprezentare compacta a sunetului, apropiata de perceptia umana
                mfccs_mean = mfccs.mean(axis=1) # calculam media valorilor mfcc extrase
                beat_strength = np.mean(librosa.onset.onset_strength(y=y, sr=sr)) # puterea batailor
                tempogram = librosa.feature.tempogram(y=y, sr=sr)
                tempo_variation = np.std(tempogram) # deviatia tempo-ului
                onsets = librosa.onset.onset_detect(y=y, sr=sr)
                onset_density = len(onsets) / librosa.get_duration(y=y, sr=sr) # densitatea evenimentelor sonore

                feature = {
                    'filename': file,
                    'genre': genre,
                    'tempo': tempo_val,
                    'spectral_centroid': spectral_centroid,
                    'zero_crossing_rate': zero_crossing_rate,
                    'rmse': rmse,
                    'bandwidth': bandwidth,
                    'chroma': chroma,
                    'beat_strength': beat_strength,
                    'tempo_variation': tempo_variation,
                    'onset_density': onset_density
                }

                # adaugam individual densitatile pentru fiecare valoare din mfcc
                for i in range(13):
                    feature[f"mfcc{i+1}"] = mfccs_mean[i]

                features.append(feature)

            except Exception as e:
                print(f"[Eroare la {file}]: {e}")

# cream csv-ul care va fi utilizat de catre programul PySpark
df = pd.DataFrame(features)
df.to_csv('features/genre_features.csv', index=False)
