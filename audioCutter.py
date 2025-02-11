import os
import tkinter as tk
from tkinter import filedialog
from pydub import AudioSegment
from pydub.utils import make_chunks

def browse_file():
    file_path.set(filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav")]))
    
def cut_audio():
    input_file = file_path.get()
    if not input_file:
        return

    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    cut_interval = int(cut_interval_var.get())
    audio = AudioSegment.from_file(input_file)
    chunks = make_chunks(audio, cut_interval * 1000)
    
    for i, chunk in enumerate(chunks):
        output_file = os.path.join(output_dir, f"chunk-{i}.{audio.format}")
        chunk.export(output_file, format=audio.format)

    result_label.config(text=f"Audio cut into {len(chunks)} segments")

root = tk.Tk()
root.title("Audio Cutter")

file_path = tk.StringVar()
cut_interval_var = tk.StringVar()

frame1 = tk.Frame(root)
frame1.pack(padx=10, pady=10)
frame2 = tk.Frame(root)
frame2.pack(padx=10, pady=10)
frame3 = tk.Frame(root)
frame3.pack(padx=10, pady=10)

tk.Label(frame1, text="Audio File:").pack(side=tk.LEFT)
tk.Entry(frame1, textvariable=file_path, width=40).pack(side=tk.LEFT)
tk.Button(frame1, text="Browse", command=browse_file).pack(side=tk.LEFT)

tk.Label(frame2, text="Cut Interval (seconds):").pack(side=tk.LEFT)
tk.Entry(frame2, textvariable=cut_interval_var, width=10).pack(side=tk.LEFT)

tk.Button(frame3, text="Cut Audio", command=cut_audio).pack(side=tk.LEFT)
result_label = tk.Label(frame3, text="")
result_label.pack(side=tk.LEFT)

root.mainloop()