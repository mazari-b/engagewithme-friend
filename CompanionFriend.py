import cv2
from eye_monitoring import EyeMonitoring 
import speech_recognition as sr
import pyttsx3
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import tkinter as tk
from tkinter import Button, filedialog
from ttkthemes import ThemedStyle
import imageio
from PIL import Image, ImageTk
import threading
import time
from queue import Queue

# Initialize the recognizer and TTS engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Load conversational AI model and tokenizer
model_name = "microsoft/DialoGPT-medium" #"microsoft/DialoGPT-medium"  # You can switch to another model if desired
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Initialize the GUI window
window = tk.Tk()
window.title("AI Bear")
window.geometry("590x530")  # Adjust window size to fit traffic light

# TTK window theme
style = ThemedStyle(window)
style.set_theme("arc")  # Choose a theme, e.g., "arc"

# Load traffic light images
red_light_image = Image.open("eye_monitoring/gui_files/red-light.png").resize((36, 53), Image.LANCZOS)
green_light_image = Image.open("eye_monitoring/gui_files/green-light.png").resize((36, 53), Image.LANCZOS)
red_light_photo = ImageTk.PhotoImage(red_light_image)
green_light_photo = ImageTk.PhotoImage(green_light_image)

# Create label to display video frames
video_label = tk.Label(window)
video_label.pack(expand=True, fill="both")

# Create label for traffic light
traffic_light_label = tk.Label(window, image=red_light_photo, bg=window.cget('bg'))
#traffic_light_label.image = red_light_photo
traffic_light_label.place(relx=0.5, rely=0.05, anchor="center")
traffic_light_label.tkraise()

# Create label to display gaze feedback
#eye_feedback_label = tk.Label(window, font=("Helvetica", 24, "bold"))
#eye_feedback_label.place(relx=0.5, rely=0.1, anchor="center")

# Flag to control video looping and threading queue
stop_looping = threading.Event()
video_queue = Queue()

# Eye tracking initialization
eyeMonitoring = EyeMonitoring()
webcam = cv2.VideoCapture(0)

# Variables for eye contact score
eye_contact_duration = 0
total_duration = 0
eye_tracking_active = False

# Transcript storage
transcript = []

def play_video(video_path, loop=False):
    def stream():
        global stop_looping
        stop_looping.clear()
        #print(f"Starting video: {video_path}")
        reader = imageio.get_reader(video_path)
        while not stop_looping.is_set():
            for frame in reader:
                if stop_looping.is_set():
                    reader.close()
                    #print(f"Stopping video: {video_path}")
                    return
                frame_image = ImageTk.PhotoImage(Image.fromarray(frame).resize((640, 480), Image.LANCZOS))
                video_queue.put(frame_image)
                time.sleep(1 / reader.get_meta_data()['fps'])
            if not loop:
                break
        reader.close()
        #print(f"Video ended: {video_path}")

    stop_looping.set()
    threading.Thread(target=stream).start()

def update_video_frame():
    if not video_queue.empty():
        frame_image = video_queue.get()
        video_label.config(image=frame_image)
        video_label.image = frame_image
    window.after(20, update_video_frame)

def stop_video():
    stop_looping.set()
    time.sleep(0.1)  # Short delay to ensure the video thread stops

def listen_to_microphone():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

def generate_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    response_ids = model.generate(inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    print(f"AI: {response}")
    return response

def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def calculate_eye_contact_score():
    global eye_contact_duration, total_duration
    if total_duration == 0:
        return 0
    score = (eye_contact_duration / total_duration) * 100
    return min(max(int(score), 0), 100)

def update_traffic_light(status):
    if status == 'contact':
        traffic_light_label.config(image=green_light_photo)
    elif status == 'no_contact':
        traffic_light_label.config(image=red_light_photo)

def handle_conversation():
    global eye_contact_duration, total_duration, eye_tracking_active

    # Stop current video and play transition video
    stop_video()
    play_video("eye_monitoring/gui_files/bear_to_listening.mp4", loop=False)
    time.sleep(2)  # Ensure the video switches to bear_listening.mp4 before listening
    play_video("eye_monitoring/gui_files/bear_is_listening.mp4", loop=True)  # Play listening loop

    eye_contact_duration = 0
    total_duration = 0
    eye_tracking_active = True

    user_input = listen_to_microphone()

    eye_tracking_active = False

    if user_input:
        stop_video()
        eye_contact_score = calculate_eye_contact_score()
        response = generate_response(user_input)

        # Append to transcript
        transcript.append(f"User: {user_input} (Eye contact score: {eye_contact_score})")
        transcript.append(f"AI: {response} (Eye contact score: {eye_contact_score})")

        play_video("eye_monitoring/gui_files/bear_stop_listening.mp4", loop=False)
        time.sleep(2)  # Ensure the video switches to bear_stop_listening.mp4
        play_video("eye_monitoring/gui_files/bear_speaking.mp4", loop=True)  # Play speaking loop
        speak_text(response)
    stop_video()
    play_video("eye_monitoring/gui_files/bear_idle.mp4", loop=True)

def on_speak_button_click():
    threading.Thread(target=handle_conversation).start()

def eye_monitoring():
    global eye_contact_duration, total_duration, eye_tracking_active
    while True:
        _, frame = webcam.read()
        eyeMonitoring.update(frame)

        if eye_tracking_active:
            if eyeMonitoring.looking_at_centre():
                eye_contact_duration += 1 / 30  # assuming 30 FPS
                #eye_feedback_label.config(text="YES", fg="green")
                print("Looking at centre")
                update_traffic_light('contact')
            else:
                #eye_feedback_label.config(text="NO", fg="red")
                print("Not looking at centre")
                update_traffic_light('no_contact')
            total_duration += 1 / 30  # assuming 30 FPS

        if cv2.waitKey(1) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

def save_transcript():
    # Prompt user to save transcript
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, "w") as file:
            for line in transcript:
                file.write(line + "\n")
        print(f"Transcript saved to {file_path}")

# Import speak button
button_image = Image.open("eye_monitoring/gui_files/button_speak.png")
button_photo = ImageTk.PhotoImage(button_image)

# Import stop button
stop_image = Image.open("eye_monitoring/gui_files/button_stop.png")
stop_photo = ImageTk.PhotoImage(stop_image)

# Create Speak button
speak_button = Button(window, image=button_photo, command=on_speak_button_click, borderwidth=0, highlightthickness=0)
speak_button.pack(side="bottom")

# Load your custom button image for the "Save Transcript" button
save_button_image = Image.open("eye_monitoring/gui_files/save-data.png")
save_button_photo = ImageTk.PhotoImage(save_button_image)

# Create Save Transcript button using the custom image (top right)
save_button = Button(window, image=save_button_photo, command=save_transcript, borderwidth=0, highlightthickness=0)
save_button.place(relx=0.9, rely=0.1, anchor="center")

# Play idle video initially
play_video("eye_monitoring/gui_files/bear_idle.mp4", loop=True)

# Start updating video frames
update_video_frame()

# Start eye tracking in a separate thread
eye_thread = threading.Thread(target=eye_monitoring)
eye_thread.start()

# Main loop
window.mainloop()
