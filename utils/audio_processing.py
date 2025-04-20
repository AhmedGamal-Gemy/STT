import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
from typing import Optional
import wave

class AudioProcessor:
    
    def __init__(
            self,
            sample_rate : int = 16000,      # Sample rate is how many samples taken per second as the audio is continous (infinite) so we just take samples
            n_mfcc : int = 13,              # MFCC is excatly like PCA It compress the audio to number of MFCCs 
            frame_length: int = 2048,       # Frame length is how much sample is being processed at a time
            hop_length: int = 512,          # Hop length is the distance ( in samples ) between the starting point of one frame and the starting point of the next frame
            speed_factors=(0.9, 1.0, 1.1)   # Speed factors for which speeding the audio (Data agumentation)
            ):
        
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.speed_factors = speed_factors
        self.audio_interface = pyaudio.PyAudio()

    def speed_audio(self, audio):
        # Add speed perturbation without changing audio length
        factor = np.random.choice(self.speed_factors)
        if factor == 1.0:
            return audio
            
        # Apply speed change
        perturbed = librosa.effects.time_stretch(audio, rate=factor)
        
        # Maintain original length
        if len(perturbed) < len(audio):
            return np.pad(perturbed, (0, len(audio) - len(perturbed)))
        return perturbed[:len(audio)]
    # Function for recording audio and then return the audio as multi diminsional numpy array
    def record_audio(self, duration : float = 1.0) -> np.ndarray :

        stream = self.audio_interface.open(
            rate= self.sample_rate,
            channels=1, # 1 -> Mono the sound is coming from one direction, 2 -> Steoro from two different directions
            format= pyaudio.paFloat32,
            input=True,
            frames_per_buffer= self.frame_length
        )

        print(f"Recording {duration} seconds....")

        frames = []

        # This append each frame from the stream to the list and this by 
        # calcuating the whole number of samples that will be processed (sample_rate * duration) then 
        # divide it by ( frame length ) so we got total number of frames needed
        for _ in range(0, int( self.sample_rate * duration / self.frame_length  ) ): 
            
            frame_buffer = stream.read(self.frame_length)
            
            # We should convert the frame buffer from the stream to numpy array
            frames.append( np.frombuffer( frame_buffer, dtype= np.float32 ) )

        # Closing everything and return the audio array

        stream.stop_stream()
        stream.close()

        print("Recording completed")

        output_file = wave.open("test.wav","wb")

        output_file.setframerate(self.sample_rate)
        output_file.setnchannels(1)
        output_file.setsampwidth( self.audio_interface.get_sample_size(pyaudio.paFloat32) )
        output_file.writeframes( b''.join(frames) )

        output_file.close()

        return np.concatenate(frames)

    # Converting audio to number of mfcc ( Exactly like number of components in PCA ) and time steps and return tensor 
    def audio_to_mfcc(self, audio : np.ndarray) -> tf.Tensor:
        
        # Make sure the type is float 32
        audio = audio.astype(np.float32)
        
        # Speed the audio
        audio = self.speed_audio(audio)
        
        # Normalize the audio
        audio = audio / ( np.max( np.abs(audio) ) + 1e-12)

        mfcc = librosa.feature.mfcc(
            y = audio,
            sr = self.sample_rate,
            n_mfcc = self.n_mfcc,
            n_fft = self.frame_length,
            hop_length = self.hop_length
        )
        # Instead of returning the n_mfcc features as rows and time steps as column. Transpose them.
        return tf.convert_to_tensor( mfcc.T, dtype = tf.float32 ) # (time steps, n_mfcc)

    # Visualize the raw audio waveform
    def plot_audio(self, audio : np.ndarray, title : str = "WaveForm") -> None :

        plt.figure( figsize=(12,3) )
        librosa.display.waveshow(y = audio, sr = self.sample_rate )
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

    # Visualize the MFCC
    def plot_mfcc(self, mfcc: tf.Tensor, title : str = "MFCC") -> None :

        # Converting the tensor to numpy ( for visualiztion ) and transpose it because of the librosa function 
        mfcc_np = mfcc.numpy().T
        
        plt.figure( figsize= (12,4) )

        librosa.display.specshow(
            mfcc_np,
            sr = self.sample_rate,
            hop_length = self.hop_length,
            x_axis = "time",
            y_axis = "mel"
        )

        plt.colorbar(format="%+2.0f dB")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    # Padding the audio to specified length
    def pad_audio(
            self,
            audio : np.ndarray,
            target_length : Optional[int] = None
    ) -> np.ndarray :
        
        if target_length is None:
            # Use Common Voice's max duration 10 sec
            target_length = 10 * self.sample_rate
            target_length = int(target_length * 1.2) # Add safety margin to target length
            
        if len(audio) > target_length:
            return audio[:target_length]
        
        return np.pad(audio, (0, target_length - len(audio)))

if __name__ == "__main__" :
    
    # Initlize the processor
    processor = AudioProcessor()

    # record and visualize the audio
    audio = processor.record_audio()
    processor.plot_audio(audio=audio)

    # Process and visualize the MFCC
    mfcc = processor.audio_to_mfcc(audio)
    print("MFCC shape:", mfcc.shape)
    processor.plot_mfcc(mfcc)

    # Testing padding
    padded = processor.pad_audio(audio,5)
    print(f"Orginal length: {len(audio)}, Padded length: {len(padded)} ")


