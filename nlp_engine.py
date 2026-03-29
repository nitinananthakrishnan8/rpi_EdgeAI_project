import os
import time
import services

class NLPEngine:
    def __init__(self):
        self.previous_sign = None
        self.last_spoken_time = 0
        self.cooldown = 4.0
        self.semantic_dict = {
            "Hello": "Hello! I am using an Edge AI system.",
            "Yes": "Yes, I agree.",
            "Thumbsup": "I am doing good, everything is okay.",
            "Pointing": "I would like that item, please.",
            "Raised": "I have a question or need attention.",
            "Pinch": "Please adjust this setting.",
            "Call": "I need to make a phone call.",
            "Peace": "I am at peace, no problems.",
            "L": "I need emergency assistance now."
        }
        self.context_dict = {
            ("Pointing", "Yes"): "Yes, that is exactly the one I want.",
            ("Hello", "L"): "Hello, this is an emergency! Please help me.",
            ("Hello", "Thumbsup"): "TRIGGER_API" # Handled in logic below
        }

    def speak_text(self, text):
        # Bluetooth-safe Asynchronous Pipeline
        temp_wav = f"/tmp/s_{int(time.time())}.wav"
        cmd = f'flite -t "{text}" -o {temp_wav} && pw-play {temp_wav} && rm {temp_wav} &'
        os.system(cmd)

    def process_and_speak(self, current_sign):
        if (time.time() - self.last_spoken_time) < self.cooldown: return
        
        sentence = ""
        # Context Check
        if self.previous_sign == "Hello" and current_sign == "Thumbsup":
            sentence = services.get_daily_briefing()
            self.previous_sign = None
        elif self.previous_sign is not None and (self.previous_sign, current_sign) in self.context_dict:
            sentence = self.context_dict[(self.previous_sign, current_sign)]
            self.previous_sign = None
        else:
            sentence = self.semantic_dict.get(current_sign, "")
            self.previous_sign = current_sign

        if sentence:
            self.speak_text(sentence)
            self.last_spoken_time = time.time()
            return sentence
        return None
