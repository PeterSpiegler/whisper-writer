import time
import threading
import traceback
import numpy as np
import sounddevice as sd
import tempfile
import wave
import webrtcvad
from PyQt5.QtCore import QThread, QMutex, pyqtSignal
from collections import deque
from threading import Event

from transcription import transcribe
from utils import ConfigManager


class ResultThread(QThread):
    """
    A thread class for handling audio recording, transcription, and result processing.

    This class manages the entire process of:
    1. Recording audio from the microphone
    2. Detecting speech and silence
    3. Saving the recorded audio as numpy array
    4. Transcribing the audio
    5. Emitting the transcription result

    In streaming mode, transcription happens periodically during recording,
    emitting intermediate results for real-time text output.

    Signals:
        statusSignal: Emits the current status of the thread (e.g., 'recording', 'transcribing', 'idle')
        resultSignal: Emits the final transcription result
        intermediateSignal: Emits intermediate transcription during streaming (full text so far)
    """

    statusSignal = pyqtSignal(str)
    resultSignal = pyqtSignal(str)
    intermediateSignal = pyqtSignal(str)

    def __init__(self, local_model=None):
        """
        Initialize the ResultThread.

        :param local_model: Local transcription model (if applicable)
        """
        super().__init__()
        self.local_model = local_model
        self.is_recording = False
        self.is_running = True
        self.sample_rate = None
        self.mutex = QMutex()

        # Streaming state
        self._recording_data = []
        self._recording_lock = threading.Lock()
        self._streaming_done = threading.Event()
        self._speech_detected_event = threading.Event()

    def stop_recording(self):
        """Stop the current recording session."""
        self.mutex.lock()
        self.is_recording = False
        self.mutex.unlock()

    def stop(self):
        """Stop the entire thread execution."""
        self.mutex.lock()
        self.is_running = False
        self.mutex.unlock()
        self._streaming_done.set()
        self.statusSignal.emit('idle')
        self.wait()

    def run(self):
        """Main execution method for the thread."""
        try:
            if not self.is_running:
                return

            self.mutex.lock()
            self.is_recording = True
            self.mutex.unlock()

            streaming = ConfigManager.get_config_value('recording_options', 'streaming')

            self.statusSignal.emit('recording')
            ConfigManager.console_print('Recording...')

            if streaming:
                self._run_streaming()
            else:
                self._run_batch()

        except Exception as e:
            traceback.print_exc()
            self.statusSignal.emit('error')
            self.resultSignal.emit('')
        finally:
            self.stop_recording()

    def _run_batch(self):
        """Original batch transcription flow."""
        audio_data = self._record_audio()

        if not self.is_running:
            return

        if audio_data is None:
            self.statusSignal.emit('idle')
            return

        self.statusSignal.emit('transcribing')
        ConfigManager.console_print('Transcribing...')

        start_time = time.time()
        result = transcribe(audio_data, self.local_model)
        end_time = time.time()

        transcription_time = end_time - start_time
        ConfigManager.console_print(f'Transcription completed in {transcription_time:.2f} seconds. Post-processed line: {result}')

        if not self.is_running:
            return

        self.statusSignal.emit('idle')
        self.resultSignal.emit(result)

    def _run_streaming(self):
        """Streaming transcription: transcribe periodically during recording."""
        recording_options = ConfigManager.get_config_section('recording_options')
        interval_ms = recording_options.get('streaming_interval') or 2000
        interval_s = interval_ms / 1000.0

        # Reset streaming state
        self._recording_data = []
        self._recording_lock = threading.Lock()
        self._streaming_done = threading.Event()
        self._speech_detected_event = threading.Event()

        # Start transcription worker thread
        worker = threading.Thread(target=self._streaming_worker, args=(interval_s,), daemon=True)
        worker.start()

        # Record audio (stores in self._recording_data via thread-safe access)
        audio_data = self._record_audio(streaming=True)

        # Stop worker and wait for it to finish
        self._streaming_done.set()
        worker.join(timeout=30)

        if not self.is_running or audio_data is None:
            self.statusSignal.emit('idle')
            self.resultSignal.emit('')
            return

        # Final transcription of complete audio
        self.statusSignal.emit('transcribing')
        start_time = time.time()
        result = transcribe(audio_data, self.local_model, final=True)
        end_time = time.time()
        ConfigManager.console_print(f'Final transcription in {end_time - start_time:.2f}s: {result}')

        if not self.is_running:
            return

        self.statusSignal.emit('idle')
        self.resultSignal.emit(result)

    def _streaming_worker(self, interval):
        """Worker thread that periodically transcribes accumulated audio."""
        # Wait for speech to be detected before starting transcription
        while not self._streaming_done.is_set():
            if self._speech_detected_event.wait(timeout=0.5):
                break

        if self._streaming_done.is_set():
            return

        while not self._streaming_done.wait(timeout=interval):
            # Get snapshot of current audio
            with self._recording_lock:
                if not self._recording_data:
                    continue
                snapshot = np.array(self._recording_data, dtype=np.int16)

            duration = len(snapshot) / (self.sample_rate or 16000)
            if duration < 0.5:
                continue

            try:
                result = transcribe(snapshot, self.local_model, final=False)
                if result:
                    ConfigManager.console_print(f'Streaming transcription: {result}')
                    self.intermediateSignal.emit(result)
            except Exception as e:
                ConfigManager.console_print(f'Streaming transcription error: {e}')

    def _record_audio(self, streaming=False):
        """
        Record audio from the microphone and save it to a temporary file.

        :param streaming: If True, store audio in thread-safe self._recording_data
        :return: numpy array of audio data, or None if the recording is too short
        """
        recording_options = ConfigManager.get_config_section('recording_options')
        self.sample_rate = recording_options.get('sample_rate') or 16000
        frame_duration_ms = 30  # 30ms frame duration for WebRTC VAD
        frame_size = int(self.sample_rate * (frame_duration_ms / 1000.0))
        silence_duration_ms = recording_options.get('silence_duration') or 900
        silence_frames = int(silence_duration_ms / frame_duration_ms)

        # 150ms delay before starting VAD to avoid mistaking the sound of key pressing for voice
        initial_frames_to_skip = int(0.15 * self.sample_rate / frame_size)

        # Create VAD only for recording modes that use it
        recording_mode = recording_options.get('recording_mode') or 'continuous'
        vad = None
        if recording_mode in ('voice_activity_detection', 'continuous'):
            vad = webrtcvad.Vad(2)  # VAD aggressiveness: 0 to 3, 3 being the most aggressive
            speech_detected = False
            silent_frame_count = 0

        audio_buffer = deque(maxlen=frame_size)
        recording = []

        data_ready = Event()

        def audio_callback(indata, frames, time, status):
            if status:
                ConfigManager.console_print(f"Audio callback status: {status}")
            audio_buffer.extend(indata[:, 0])
            data_ready.set()

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16',
                            blocksize=frame_size, device=recording_options.get('sound_device'),
                            callback=audio_callback):
            while self.is_running and self.is_recording:
                data_ready.wait()
                data_ready.clear()

                if len(audio_buffer) < frame_size:
                    continue

                # Save frame
                frame = np.array(list(audio_buffer), dtype=np.int16)
                audio_buffer.clear()

                if streaming:
                    with self._recording_lock:
                        self._recording_data.extend(frame)
                else:
                    recording.extend(frame)

                # Avoid trying to detect voice in initial frames
                if initial_frames_to_skip > 0:
                    initial_frames_to_skip -= 1
                    continue

                if vad:
                    if vad.is_speech(frame.tobytes(), self.sample_rate):
                        silent_frame_count = 0
                        if not speech_detected:
                            ConfigManager.console_print("Speech detected.")
                            speech_detected = True
                            if streaming:
                                self._speech_detected_event.set()
                    else:
                        silent_frame_count += 1

                    if speech_detected and silent_frame_count > silence_frames:
                        break
                elif streaming and not self._speech_detected_event.is_set():
                    # No VAD in this recording mode (press_to_toggle/hold_to_record)
                    # Start streaming transcription after initial frames
                    self._speech_detected_event.set()

        if streaming:
            with self._recording_lock:
                audio_data = np.array(self._recording_data, dtype=np.int16)
        else:
            audio_data = np.array(recording, dtype=np.int16)

        duration = len(audio_data) / self.sample_rate

        ConfigManager.console_print(f'Recording finished. Size: {audio_data.size} samples, Duration: {duration:.2f} seconds')

        min_duration_ms = recording_options.get('min_duration') or 100

        if (duration * 1000) < min_duration_ms:
            ConfigManager.console_print(f'Discarded due to being too short.')
            return None

        return audio_data
