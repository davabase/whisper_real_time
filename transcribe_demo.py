#! python3.7

import io
import os
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from system_configuration import ParserValues, AudioDeviceConfiguration
from audio_util import AudioUtil

class SpeechHandler:
    def __init__(self):
        self.args = ParserValues.fromSystemArguments()
        # The last time a recording was retreived from the queue.
        self.phrase_time = None
        # Current raw audio bytes.
        self.last_sample = bytes()
        # Thread safe Queue for passing data from the threaded recording callback.
        self.data_queue = Queue()
        # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.args.energy_threshold
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
        self.recorder.dynamic_energy_threshold = False

        # Important for linux users. 
        # Prevents permanent application hang and crash by using the wrong Microphone
        self.device_index = AudioDeviceConfiguration.get_microphone_device_index(self.args.default_microphone)

        # Load / Download model
        self.audio_model = self.load_mode()

        self.record_timeout = self.args.record_timeout
        self.silence_timeout = self.args.silence_timeout

        self.temp_file = NamedTemporaryFile().name

        self.generate_audio_source()

    def load_mode(self):
        args = self.args
        ONLY_ENGLISH = False
        model = args.model
        if args.model != "large" and not args.non_english and ONLY_ENGLISH:
            model = model + ".en"
        return whisper.load_model(model)

    def generate_audio_source(self):
        self.source = sr.Microphone(sample_rate=16000,device_index=self.device_index)
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        def record_callback(_, audio:sr.AudioData) -> None:
            """
            Threaded callback function to recieve audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            # Grab the raw bytes and push it into the thread safe queue.
            data = audio.get_raw_data()
            self.data_queue.put(data)

        # Create a background thread that will pass us raw audio bytes.
        # We could do this manually but SpeechRecognizer provides a nice helper.
        self.recorder.listen_in_background(self.source, record_callback, phrase_time_limit=self.record_timeout)

    def execute(self):
        # Cue the user that we're ready to go.
        print("Model loaded.\n")
        #clear terminal
        self.transcription = ['']
        is_speaking = False
        while True:
            try:
                # Pull raw recorded audio from the queue.
                if not self.data_queue.empty():
                    # If enough time has passed between recordings, consider the phrase complete.
                    # Clear the current working audio buffer to start over with the new data.
                    has_silence_timeout = self.silence_time_is_up()
                    if(has_silence_timeout): self.last_sample = bytes()

                    # This is the last time we received new audio data from the queue.
                    is_speaking = True
                    self.phrase_time = datetime.utcnow()

                    # Concatenate our current audio data with the latest audio data.
                    self.last_sample = AudioUtil.concat_data_to_current_audio(self.last_sample,self.data_queue)

                    # Use AudioData to convert the raw data to wav data.
                    audio_data = sr.AudioData(self.last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                    wav_data = io.BytesIO(audio_data.get_wav_data())

                    # Write wav data to the temporary file as bytes.
                    AudioUtil.write_temp_audio_file(self.temp_file,wav_data)

                    # Read the transcription.
                    result = self.audio_model.transcribe(self.temp_file, fp16=torch.cuda.is_available())
                    self.transcription = self.result_transcription_handler(result,has_silence_timeout)
                    self.show_transcription()
                        
                else:
                    if(is_speaking and self.silence_time_is_up()):
                        self.transcription[-1] = f"[Final]: {self.transcription[-1]}"
                        self.show_transcription()
                        is_speaking = False

            except KeyboardInterrupt:
                break
            # Infinite loops are bad for processors, must sleep.
            sleep(0.25)

        print("\n\nTranscription:")
        for line in self.transcription:
            print(line)

    def silence_time_is_up(self):
        silence_timeout = self.silence_timeout
        phrase_time = self.phrase_time
        if(phrase_time is None): return False
        now = datetime.utcnow()
        elapsed_time_delta = now - phrase_time
        has_silence_timeout = phrase_time and elapsed_time_delta > timedelta(seconds=silence_timeout)
        return has_silence_timeout

    def result_transcription_handler(self,result,has_silence_timeout):
        text = result['text'].strip()
        # If we detected a pause between recordings, add a new item to our transcripion.
        # Otherwise edit the existing one.
        if has_silence_timeout:
            self.transcription.append(text)
        else:
            self.transcription[-1] = text
        return self.transcription

    def show_transcription(self):
        # Clear the console to reprint the updated transcription.
        os.system('cls' if os.name=='nt' else 'clear')
        for line in self.transcription:
            print(line)
        # Flush stdout.
        print('', end='', flush=True)


if __name__ == "__main__":
    speechHandler = SpeechHandler()
    speechHandler.execute()
