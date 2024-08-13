import speech_recognition as sr
import pyttsx3
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Initialize the recognizer and TTS engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Load DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

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

# Main conversation loop
if __name__ == "__main__":
    print("Start talking to the AI. Say 'exit' to end the conversation.")
    while True:
        user_input = listen_to_microphone()
        if user_input is None:
            continue
        if user_input.lower() == "exit":
            break
        response = generate_response(user_input)
        speak_text(response)
