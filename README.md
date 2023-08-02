# Real-Time Whisper Transcription

![Demo gif](demo.gif)

This is a module of real-time speech-to-text with OpenAI's Whisper model. It works by constantly recording audio in a thread and concatenating the raw bytes over multiple recordings.

# Quick start
1. Clone this repository into the project folder (or $PYTHONPATH)
```
git clone https://github.com/DDadeA/whisper_real_time.git
cd whisper_real_time
pip install -r requirements.txt
```

2. Whisper requires [`ffmpeg`](https://www.ffmpeg.org/download.html).
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

3. Import module
```Python
# Load module
from time import sleep
import whisper_real_time

# Load Whisper Model
whisper = whisper_real_time.WhisperRecognizer('deemboi/whisper-small-kr') # hugging face name

while True:
  # Get transcription
  transcription = whisper.get_sentence()
  
  # is there any new sentence?
  if (transcription == None):  # no
      sleep(0.5)
      continue
  
  # Yes
  print(transcription)
```

This module will continue to operate in the background, and will continue to record until sentences are called to get_sentence().

If there is no new sentences, get_sentence() returns None.