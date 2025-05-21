from tkinter import *
from PIL import ImageTk, Image
from tkinter import ttk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import os
import subprocess
import librosa
import numpy as np
from scipy.io import wavfile
from tensorflow.keras.models import load_model


song_path=''
spleeter_output_path = 'Spleeter_Output'
chords_output_path = './Chords_Output'
model_path = './saved_models/v3.keras'
labels = ['A', 'Am', 'C', 'D', 'E', 'F', 'G']

def sup(*funcs):
    def combined_func(*args,**kwargs):
        for f in funcs:
            f(*args,**kwargs)
    return combined_func

def generate_chords():
    print("Checking if song folder exists under Spleeter_Output...")
    song_folder_path = f"{spleeter_output_path}/{songname}"

    if os.path.exists(song_folder_path):
        print(f"Using existing 'other.wav' from {song_folder_path}...")
        global other_wav_path
        other_wav_path = f"{song_folder_path}/other.wav"
    else:
        print("Separating audio into components using Spleeter...")
        subprocess.run(['spleeter', 'separate', '-p', 'spleeter:5stems', '-o', spleeter_output_path, song_path])
        other_wav_path = f"{song_folder_path}/other.wav"

    print("Extracting individual chords from other.wav...")
    y, sr = librosa.load(other_wav_path, sr=None)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    onset_env = librosa.onset.onset_strength(y=y_harmonic, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='frames', backtrack=True, hop_length=512)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    def merge_close_onsets(onset_frames, min_distance=3):
        merged_onsets = []
        last_onset = -min_distance
        for onset in onset_frames:
            if onset - last_onset > min_distance:
                merged_onsets.append(onset)
                last_onset = onset
        return np.array(merged_onsets)

    merged_onset_frames = merge_close_onsets(onset_frames, min_distance=5)
    merged_onset_times = librosa.frames_to_time(merged_onset_frames, sr=sr)

    onset_samples = librosa.frames_to_samples(onset_frames)
    segments = [y[onset_samples[i]:onset_samples[i+1]] for i in range(len(onset_samples)-1)]
    segments.append(y[onset_samples[-1]:])
    segments = [segment for segment in segments if len(segment) > 0]

    os.makedirs(chords_output_path, exist_ok=True)
    segment_files = []
    for i, segment in enumerate(segments):
        output_filename = f'{chords_output_path}/{songname}chord{i+1}.wav'
        wavfile.write(output_filename, sr, (segment * 32767).astype(np.int16))
        segment_files.append(output_filename)

    print("Predicting chords for each segment...")
    model = load_model(model_path)

    def extract_features(file):
        audio_data, sample_rate = librosa.load(file)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled 

    predicted_chords = []
    for segment_file in segment_files:
        pred_feature = extract_features(segment_file)
        pred_feature = pred_feature.reshape(1, -1)
        predicted_class_index = np.argmax(model.predict(pred_feature), axis=1)
        predicted_class_label = labels[predicted_class_index[0]]
        predicted_chords.append(predicted_class_label)
        print(f"Predicted chord for {segment_file}: {predicted_class_label}")

    display_chords(predicted_chords)

def display_chords(predicted_chords):
    chord_frame = Frame(root, width=1280, height=680, bg="black")
    chord_frame.pack(fill="both", expand=True)

    canvas = Canvas(chord_frame, width=1280, height=680, bg="black", highlightthickness=0)
    canvas.pack(side=LEFT, fill=BOTH, expand=True)

    scrollbar = Scrollbar(chord_frame, orient=VERTICAL, command=canvas.yview, bg="black")
    scrollbar.pack(side=RIGHT, fill=Y)

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    inner_frame = Frame(canvas, bg="black")
    canvas.create_window((0, 0), window=inner_frame, anchor="nw")

    title_label = Label(inner_frame, text=f"{songname} Chords", font=("Helvetica", 16), bg="black", fg="white")
    title_label.grid(row=0, column=8, columnspan=10, pady=10)

    max_chords_per_row = 20

    for i, chord in enumerate(predicted_chords):
        chord_label = Label(inner_frame, text=f"{chord}", font=("Helvetica", 14), relief="flat", padx=10, pady=5, bg="black", fg="white")
        chord_label.grid(row=1 + i // max_chords_per_row, column=i % max_chords_per_row, padx=5, pady=5)

    spacer = Frame(inner_frame, height=50, bg="black")
    spacer.grid(row=(len(predicted_chords) // max_chords_per_row) + 2, column=0, columnspan=max_chords_per_row)

    button = Button(inner_frame, text="Back to Song Choose", relief="flat", font=("Helvetica", 14), bg="white", fg="black", command=sup(chord_frame.destroy, w2))
    button.grid(row=(len(predicted_chords) // max_chords_per_row) + 3, column=6, columnspan=max_chords_per_row, pady=20)

    def _on_mousewheel(event):
        canvas.yview_scroll(-1 * (event.delta // 120), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)

def check():
    if song_path=='':
        messagebox.showerror("Error", "Please choose a song!")
    else:
        f2.destroy()
        generate_chords()

def w2():
    def open_file():
        global song_path
        song_path = askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav"), ("MP3 Files", "*.mp3"), ("WAV Files", "*.wav")])
        global songname, other_wav_path
        songname = os.path.splitext(os.path.basename(song_path))[:-1][0]
        other_wav_path = f'{spleeter_output_path}/{songname}/other.wav'
        if song_path:
            b2.config(text=os.path.basename(song_path))

    global f2
    f2 = Frame(root, width='1280', height='680')
    f2.pack(fill='both', expand=True)

    image = ImageTk.PhotoImage(Image.open("2ndpg.png"))
    lblx = Label(f2, image=image)
    lblx.image = image
    lblx.place(relx=0.0, rely=0.0)

    b2 = Button(f2, bg='black', fg='white', relief='ridge', text='Choose Song', font=('Garamond', 24), command=open_file)
    b2.place(anchor='center', relx=0.5, rely=0.56)

    b3 = Button(f2, bg='black', fg='white', text="Generate Chords", relief='flat', font=('Garamond', 28), command=check)
    b3.place(anchor='center', relx=0.5, rely=0.835)

def w1():
    f1 = Frame(root, width='1280', height='680')
    f1.pack(fill='both', expand=True)

    img = Image.open("bg1.png")
    img = img.resize((1280, 680))
    image9 = ImageTk.PhotoImage(img)
    lblx = Label(f1, image=image9)
    lblx.image = image9
    lblx.place(relx=0.0, rely=0.0)

    l = Button(f1, bg='black', fg='white', relief='flat', text="Get Started", font=('Garamond', 28), command=sup(f1.destroy, w2))
    l.place(anchor='center',relx=0.5, rely=0.848)

root = Tk()
root.title("Guitar Transcription")
root.geometry('1280x680')
root.resizable(False, False)
root.configure(bg='black')

w1()

root.mainloop()