from telegram.ext import ApplicationBuilder, MessageHandler, filters

from blinker import signal

from comms.telegram.listener.message_listener import MessageListener
from config.env_config import get_env, get_env_int_list, init_config

from constants import TELEGRAM_EVENT_SIGNAL_NAME


def on_telegram_event(sender, telegram_event, result, update, context):
    print("Signal received")
    print("sender:", sender.__class__.__name__)
    print("event:", telegram_event.model_dump() if hasattr(telegram_event, "model_dump") else telegram_event)
    print("result:", result)


def main() -> None:
    init_config()

    allowed_chat_ids = get_env_int_list("TELEGRAM_LISTENER_CHAT_ID", required=True)
    bot_token = get_env("JATAYU_TELEGRAM_TOKEN", required=True) or ""

    app = ApplicationBuilder().token(bot_token).build()
    listener = MessageListener(allowed_chat_ids)

    event_receiver = signal(TELEGRAM_EVENT_SIGNAL_NAME)
    event_receiver.connect(on_telegram_event)

    app.add_handler(MessageHandler(filters.ALL, listener.on_message))
    app.run_polling()


if __name__ == "__main__":
    main()
