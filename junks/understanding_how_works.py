import torch, whisper

audio_model = whisper.load_model('medium')
print(' > model loaded! ')

result = audio_model.transcribe('test.wav', fp16=torch.cuda.is_available())
text = result['text'].strip()
print(text)