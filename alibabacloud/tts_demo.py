# coding=utf-8
#
# Installation instructions for pyaudio:
# APPLE Mac OS X
#   brew install portaudio 
#   pip install pyaudio
# Debian/Ubuntu
#   sudo apt-get install python-pyaudio python3-pyaudio
#   or
#   pip install pyaudio
# CentOS
#   sudo yum install -y portaudio portaudio-devel && pip install pyaudio
# Microsoft Windows
#   python -m pip install pyaudio

import dashscope
import sys
import pyaudio
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
from dashscope.audio.tts import ResultCallback, SpeechSynthesizer, SpeechSynthesisResult



class Callback(ResultCallback):
    _player = None
    _stream = None

    def on_open(self):
        print('Speech synthesizer is opened.')
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16,
            channels=1, 
            rate=24000,
            output=True)

    def on_complete(self):
        print('Speech synthesizer is completed.')

    def on_error(self, response: SpeechSynthesisResponse):
        print('Speech synthesizer failed, response is %s' % (str(response)))

    def on_close(self):
        print('Speech synthesizer is closed.')
        self._stream.stop_stream()
        self._stream.close()
        self._player.terminate()

    def on_event(self, result: SpeechSynthesisResult):
        if result.get_audio_frame() is not None:
            print('audio result length:', sys.getsizeof(result.get_audio_frame()))
            self._stream.write(result.get_audio_frame())

        if result.get_timestamp() is not None:
            print('timestamp result:', str(result.get_timestamp()))

callback = Callback()
SpeechSynthesizer.call(model='sambert-zhichu-v1',
                       text='你好，我是无限引擎AI小助手，可以回答您的问题',
                       sample_rate=24000,
                       format='pcm',
                       callback=callback)