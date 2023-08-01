# Real-Time Whisper Transcription

![Demo gif](demo.gif)

This is a demo of real-time speech-to-text with OpenAI's Whisper model. It works by constantly recording audio in a thread and concatenating the raw bytes over multiple recordings.

To install dependencies simply run
```
pip install -r requirements.txt
```
in an environment of your choosing.

Whisper also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

For more information on Whisper please see https://github.com/openai/whisper

The code in this repository is public domain.


## transcribe_pack.py Example
You can use this module for simplicity.
```Python
# Load module
from time import sleep
import transcribe_pack

# Load Whisper Model
'''
For large, medium, small ko_whisper model, use
model_name:int = [0, 1, 2] 

For a custom huggingface model, use
model_name:str
'''
model = transcribe_pack.WhisperRecognizer(model_name='deemboi/whisper-small-kr') 

while True:
  # Get data
  result = model.get_sentence()
  
  # is there any new sentence?
  if (result == None):  # no
      sleep(0.5)
      continue
  
  # Yes
  print(result)
```
