import librosa
import numpy as np
import soundfile as sf
import os

def augment_audio(audio, sr):
    augmented_audio = audio.copy()

    shift_range = int(sr * 0.2)
    shift = np.random.randint(-shift_range, shift_range)
    augmented_audio = np.roll(augmented_audio, shift)

    # Pitch shifting
    pitch_shift_range = 2
    pitch_shift = np.random.randint(-pitch_shift_range, pitch_shift_range)
    augmented_audio = librosa.effects.pitch_shift(y=augmented_audio, sr=sr, n_steps=pitch_shift)

    # Time stretching
    time_stretch_range = 0.2
    time_stretch = 1 + np.random.uniform(-time_stretch_range, time_stretch_range)
    augmented_audio = librosa.effects.time_stretch(y=augmented_audio, rate=time_stretch)

    return augmented_audio


input_folder = "V3/Training/C" ###### To create dataset for different chord, change chord letter F here
output_folder = "AugV3/C" ###### and here
# Done until C
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        file_path = os.path.join(input_folder, filename)
        audio, sr = librosa.load(file_path)
        
        for i in range(10):
            augmented_audio = augment_audio(audio, sr)

            output_filename = f"{filename}_{i}.wav"
            output_path = os.path.join(output_folder, output_filename)

            sf.write(output_path, augmented_audio, sr)