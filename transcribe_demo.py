#! python3.7

import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform


def parser_validation(parser):
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)  
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()
    return args

def get_microphone_device_index(mic_name):
    #If is not a linux system, then return None
    if not 'linux' in platform:
        return None
    #If is requesting fot the list, print it and exit the program
    if not mic_name or mic_name == 'list':
        print("Available microphone devices are: ")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Microphone with name \"{name}\" found")   
        exit()
    #If non of the above, then return the microphone found or None
    device_index = None
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        if mic_name in name:
            device_index = index
            break
    return device_index

def load_mode(args):
    ONLY_ENGLISH = False
    model = args.model
    if args.model != "large" and not args.non_english and ONLY_ENGLISH:
        model = model + ".en"
    return whisper.load_model(model)

def result_transcription_handler(result,transcription,has_silence_timeout):
    text = result['text'].strip()
    # If we detected a pause between recordings, add a new item to our transcripion.
    # Otherwise edit the existing one.
    if has_silence_timeout:
        transcription.append(text)
    else:
        transcription[-1] = text
    return transcription

def show_transcription(transcription):
    # Clear the console to reprint the updated transcription.
    os.system('cls' if os.name=='nt' else 'clear')
    for line in transcription:
        print(line)
    # Flush stdout.
    print('', end='', flush=True)

def write_temp_audio_file(temp_file,wav_data):
    # Write wav data to the temporary file as bytes.
    with open(temp_file, 'w+b') as f:
        f.write(wav_data.read())

def concat_data_to_current_audio(last_sample,data_queue):
    while not data_queue.empty():
        data = data_queue.get()
        last_sample += data
    return last_sample

def silence_time_is_up(silence_timeout,phrase_time):
    now = datetime.utcnow()
    has_silence_timeout = False
    if(phrase_time is None): return has_silence_timeout
    elapsed_time_delta = now - phrase_time
    if phrase_time and elapsed_time_delta > timedelta(seconds=silence_timeout):
        has_silence_timeout = True
    return has_silence_timeout

def main():
    args = parser_validation(argparse.ArgumentParser())
    
    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False
    
    # Important for linux users. 
    # Prevents permanent application hang and crash by using the wrong Microphone
    device_index = get_microphone_device_index(args.default_microphone)
        
    # Load / Download model
    audio_model = load_mode(args)

    record_timeout = args.record_timeout
    silence_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    
    source = sr.Microphone(sample_rate=16000,device_index=device_index)
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")
    #clear terminal
    transcription = ['']
    is_speaking = False
    while True:
        try:
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                has_silence_timeout = silence_time_is_up(silence_timeout,phrase_time)
                if(has_silence_timeout): last_sample = bytes()

                # This is the last time we received new audio data from the queue.
                is_speaking = True
                phrase_time = datetime.utcnow()

                # Concatenate our current audio data with the latest audio data.
                last_sample = concat_data_to_current_audio(last_sample,data_queue)

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                write_temp_audio_file(temp_file,wav_data)

                # Read the transcription.
                result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                transcription = result_transcription_handler(result,transcription,has_silence_timeout)
                show_transcription(transcription)
                    
            else:
                if(is_speaking and silence_time_is_up(silence_timeout,phrase_time)):
                    transcription[-1] = f"[Final]: {transcription[-1]}"
                    show_transcription(transcription)
                    is_speaking = False

        except KeyboardInterrupt:
            break
        # Infinite loops are bad for processors, must sleep.
        sleep(0.25)

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
