# Libs
from time import sleep
import os

import whisper_real_time



if __name__ == '__main__':
                
    # Load Whisper Model
    whisper = whisper_real_time.WhisperRecognizer('deemboi/whisper-small-kr') # "byoussef/whisper-large-v2-Ko", "seastar105/whisper-medium-ko-zeroth", 'deemboi/whisper-small-kr'
    print('Loaded')
    
    while True:
        try:
            # Infinite loops are bad for processors, must sleep.
            sleep(0.25)

            # Get transcription
            transcription = whisper.get_sentence()
            if transcription == None: continue
            
            # Clear and Print
            os.system('cls' if os.name=='nt' else 'clear')
            
            for line in transcription:
                print(line)
            
            # Flush stdout.
            print('', end='', flush=True)
            
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()