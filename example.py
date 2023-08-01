# Load module
from time import sleep
import transcribe_pack

# Load Whisper Model
model = transcribe_pack.WhisperRecognizer(2) # [0 = large, 1 = medium, 2 = small]

while True:
  # Get data
  result = model.get_sentence()
  
  # is there any new sentence?
  if (result == None):  # no
      sleep(0.5)
      continue
  
  # Yes
  print(result)
