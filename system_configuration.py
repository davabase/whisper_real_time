import argparse
import speech_recognition as sr

from sys import platform

class ParserValues:
    model: str
    non_english: bool
    energy_threshold: int
    record_timeout: float
    silence_timeout: float
    default_microphone: str

    def __init__(self, model, non_english, energy_threshold, record_timeout, silence_timeout, default_microphone):
        self.model = model
        self.non_english = non_english
        self.energy_threshold = energy_threshold
        self.record_timeout = record_timeout
        self.silence_timeout = silence_timeout
        self.default_microphone = default_microphone

    @classmethod 
    def parser_validation(cls,parser):
        parser.add_argument("--model", default="medium", help="Model to use",
                            choices=["tiny", "base", "small", "medium", "large"])
        parser.add_argument("--non_english", action='store_true',
                            help="Don't use the english model.")
        parser.add_argument("--energy_threshold", default=1000,
                            help="Energy level for mic to detect.", type=int)
        parser.add_argument("--record_timeout", default=2,
                            help="How real time the recording is in seconds.", type=float)
        parser.add_argument("--silence_timeout", default=3,
                            help="How much empty space between recordings before we "
                                "consider it a new line in the transcription.", type=float)  
        if 'linux' in platform:
            parser.add_argument("--default_microphone", default='pulse',
                                help="Default microphone name for SpeechRecognition. "
                                    "Run this with 'list' to view available Microphones.", type=str)
        args = parser.parse_args()
        return args

    @classmethod
    def fromSystemArguments(cls):
        parser = argparse.ArgumentParser()
        args = cls.parser_validation(parser)
        return cls(
            model=args.model,
            non_english=args.non_english,
            energy_threshold=args.energy_threshold,
            record_timeout=args.record_timeout,
            silence_timeout=args.silence_timeout,
            default_microphone=args.default_microphone
        )

class AudioDeviceConfiguration:

    @staticmethod
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