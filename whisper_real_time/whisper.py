import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from subprocess import CalledProcessError, run


class WhisperModel():
    def __init__(self, model_name:str, fp16_available:bool):
        if fp16_available:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).cuda()
        else:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
            
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.fp16_available = fp16_available
        
            

    def transcribe(self, file_path:str) -> str:
        
        # Load audio file
        wavarr = load_audio(file_path)

        # Prediction
        if self.fp16_available:
            input_features = self.processor(wavarr, sampling_rate=16000, return_tensors='pt').input_features.cuda()
        else:
            input_features = self.processor(wavarr, sampling_rate=16000, return_tensors='pt').input_features

        predicted_ids = self.model.generate(input_features, max_new_tokens=64)

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        return {'text':transcription[0]}


def load_audio(file: str, sr: int = 16000):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def load_model(model_name:str, fp16:bool=False):
    return WhisperModel(model_name, fp16)