import sys
import argparse
from pathlib import Path

from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from loguru import logger

from .conversation_agent import create_agent

# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOCUMENTS_PATH = PROJECT_ROOT / "documents"

def start():
    parser = argparse.ArgumentParser(description="Customer Care Voice Agent")
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface (get a temp phone number)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of the knowledge base.",
    )
    args = parser.parse_args()

    # Configure logging
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")

    if args.reindex:
        logger.info("Vector DB status: --reindex flag detected. Forcing a full re-index of the knowledge base.")
    else:
        logger.info("Vector DB status: Using existing index. To force a re-index, use the --reindex flag.")

    # Initialize models and agent
    stt_model = get_stt_model()  # moonshine/base
    tts_model = get_tts_model()  # kokoro
    customer_care_agent = create_agent(
        documents_path=DOCUMENTS_PATH, reindex=args.reindex
    )

    def run_agent(audio):
        transcript = stt_model.stt(audio)
        logger.debug(f"🎤 Transcript: {transcript}")

        # Invoke the agent graph
        result = customer_care_agent.invoke({"question": transcript})
        response_text = result.get("response", "I am sorry, I could not find an answer.")

        logger.debug(f"🤖 Response: {response_text}")
        for audio_chunk in tts_model.stream_tts_sync(response_text):
            yield audio_chunk

    def create_stream():
        return Stream(ReplyOnPause(run_agent), modality="audio", mode="send-receive")
    stream = create_stream()

    if args.phone:
        logger.info("Launching with FastRTC phone interface...")
        stream.fastphone()
    else:
        logger.info("Launching with Gradio UI...")
        stream.ui.launch()

if __name__ == "__main__":
    start() 