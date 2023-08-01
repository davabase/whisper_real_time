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
