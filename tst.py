import asyncio

from input.sources.microphone import MicrophoneSource


async def main():
    source = MicrophoneSource()

    async for frame in source.stream():
        print(frame)
        break


if __name__ == "__main__":
    asyncio.run(main())