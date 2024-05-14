import asyncio
import contextlib
import dataclasses
import logging
import os
from dataclasses import dataclass
from typing import Any, AsyncIterable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from livekit import rtc
from livekit.agents import tts
import dashscope
import sys
from .models import TTSModels
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
from dashscope.audio.tts import ResultCallback, SpeechSynthesizer, SpeechSynthesisResult





@dataclass
class Voice:
    id: str
    name: str
    category: str
    settings: Optional["VoiceSettings"] = None


@dataclass
class VoiceSettings:
    stability: float  # [0.0 - 1.0]
    similarity_boost: float  # [0.0 - 1.0]
    style: Optional[float] = None  # [0.0 - 1.0]
    use_speaker_boost: Optional[bool] = False


DEFAULT_VOICE = Voice(
    id="EXAVITQu4vr4xnSDxMaL",
    name="Bella",
    category="premade",
    settings=VoiceSettings(
        stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True
    ),
)

API_BASE_URL_V1 = "https://api.elevenlabs.io/v1"
AUTHORIZATION_HEADER = "xi-api-key"
STREAM_EOS = ""


@dataclass
class TTSOptions:
    api_key: str
    voice: Voice
    model_id: TTSModels
    base_url: str
    sample_rate: int
    latency: int

class Callback(ResultCallback):
    def __init__(self, _tts: tts.SynthesizeStream):
        self._tts = _tts
    def on_open(self):
        print('Speech synthesizer is opened.')

    def on_complete(self):
        print('Speech synthesizer is completed.')

    def on_error(self, response: SpeechSynthesisResponse):
        print('Speech synthesizer failed, response is %s' % (str(response)))

    def on_close(self):
        print('Speech synthesizer is closed.')

    def on_event(self, result: SpeechSynthesisResult):
        audio_frame = result.get_audio_frame()
        if  audio_frame is not None:
            audio_frame = rtc.AudioFrame(
                data=audio_frame,
                sample_rate=24000,
                num_channels=1,
                samples_per_channel=len(audio_frame) // 2,
            )
            
            self._tts._event_queue.put_nowait(
                tts.SynthesisEvent(
                    type=tts.SynthesisEventType.AUDIO,
                    audio=tts.SynthesizedAudio(text="", data=audio_frame),
                )
            )

class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: Voice = DEFAULT_VOICE,
        model_id: TTSModels = "sambert-zhichu-v1",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        sample_rate: int = 24000,
        latency: int = 2,
    ) -> None:
        super().__init__(streaming_supported=True)
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY must be set")

        self._session = aiohttp.ClientSession()
        self._config = TTSOptions(
            voice=voice,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url or API_BASE_URL_V1,
            sample_rate=sample_rate,
            latency=latency,
        )


    def synthesize(
        self,
        text: str,
    ) -> AsyncIterable[tts.SynthesizedAudio]:
        voice = self._config.voice

        async def generator():
            async with self._session.post(
                f"{self._config.base_url}/text-to-speech/{voice.id}?output_format=pcm_44100",
                headers={AUTHORIZATION_HEADER: self._config.api_key},
                json=dict(
                    text=text,
                    model_id=self._config.model_id,
                    voice_settings=dataclasses.asdict(voice.settings)
                    if voice.settings
                    else None,
                ),
            ) as resp:
                data = await resp.read()
                yield tts.SynthesizedAudio(
                    text=text,
                    data=rtc.AudioFrame(
                        data=data,
                        sample_rate=44100,
                        num_channels=1,
                        samples_per_channel=len(data) // 2,  # 16-bit
                    ),
                )

        return generator()

    def stream(
        self,
    ) -> "SynthesizeStream":
        return SynthesizeStream(self._session, self._config)


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        session: aiohttp.ClientSession,
        config: TTSOptions,
    ):
        self._config = config
        self._session = session
        self._executor = ThreadPoolExecutor()
        self._queue = asyncio.Queue[str]()
        self._event_queue = asyncio.Queue[tts.SynthesisEvent]()
        self._closed = False

        self._main_task = asyncio.create_task(self._run())

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(f"elevenlabs synthesis task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)
        self._text = ""


    def push_text(self, token: str | None) -> None:
        if self._closed:
            raise ValueError("cannot push to a closed stream")

        if not token or len(token) == 0:
            return

        # TODO: Native word boundary detection may not be good enough for all languages
        # fmt: off
        splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
        # fmt: on

        self._text += token
        if token[-1] in splitters:
            self._queue.put_nowait(self._text)
            self._text = ""

    async def _run(self) -> None:
        callback = Callback(self)
      
        started = False
        try:
            text = None
            text = await self._queue.get()
            if not started:
                    self._event_queue.put_nowait(
                        tts.SynthesisEvent(type=tts.SynthesisEventType.STARTED)
                    )
                    started = True
            loop = asyncio.get_event_loop()
            loop.run_in_executor(
                self._executor,  # 假设 self._executor 是一个 ThreadPoolExecutor 实例
                lambda: SpeechSynthesizer.call(
                    model='sambert-zhichu-v1',
                    text=text,
                    sample_rate=24000,
                    format='pcm',
                    callback=callback,
                ),
            )
            self._queue.task_done()
            if text == STREAM_EOS:
                # We know 11labs is closing the stream after each request/flush
                self._event_queue.put_nowait(
                    tts.SynthesisEvent(type=tts.SynthesisEventType.FINISHED)
                )
        except asyncio.CancelledError:
                pass
        except Exception as e:
                print(e)
        # while True:
        #     try:
        #         text = None
        #         text = await self._queue.get()
        #         if not started:
        #                 self._event_queue.put_nowait(
        #                     tts.SynthesisEvent(type=tts.SynthesisEventType.STARTED)
        #                 )
        #                 started = True
        #         loop = asyncio.get_event_loop()
        #         loop.run_in_executor(
        #             self._executor,  # 假设 self._executor 是一个 ThreadPoolExecutor 实例
        #             lambda: SpeechSynthesizer.call(
        #                 model='sambert-zhichu-v1',
        #                 text=text,
        #                 sample_rate=24000,
        #                 format='pcm',
        #                 callback=callback,
        #             ),
        #         )
        #         self._queue.task_done()
        #         if text == STREAM_EOS:
        #             # We know 11labs is closing the stream after each request/flush
        #             self._event_queue.put_nowait(
        #                 tts.SynthesisEvent(type=tts.SynthesisEventType.FINISHED)
        #             )
        #             break
        #     except asyncio.CancelledError:
        #         break
        #     except Exception as e:
        #         print(e)
        #     await asyncio.sleep(200)

    
    async def flush(self) -> None:
        self._queue.put_nowait(self._text + " ")
        self._text = ""
        self._queue.put_nowait(STREAM_EOS)
        await self._queue.join()

    async def aclose(self, wait=False) -> None:
        if wait:
            logging.warning(
                "wait=True is not yet supported for ElevenLabs TTS. Closing immediately."
            )
        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    async def __anext__(self) -> tts.SynthesisEvent:
        if self._closed and self._event_queue.empty():
            raise StopAsyncIteration

        return await self._event_queue.get()



