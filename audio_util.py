class AudioUtil:

    @staticmethod
    def write_temp_audio_file(temp_file,wav_data):
        # Write wav data to the temporary file as bytes.
        with open(temp_file, 'w+b') as f:
            f.write(wav_data.read())

    @staticmethod
    def concat_data_to_current_audio(last_sample,data_queue):
        while not data_queue.empty():
            data = data_queue.get()
            last_sample += data
        return last_sample