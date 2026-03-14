from telegram.ext import ApplicationBuilder, MessageHandler, filters

from comms.telegram.listener.message_listener import MessageListener
from config.env_config import get_env, get_env_int_list, init_config


def main() -> None:
    init_config()

    allowed_chat_ids = get_env_int_list("TELEGRAM_LISTENER_CHAT_ID", required=True)
    bot_token = get_env("JATAYU_TELEGRAM_TOKEN", required=True) or ""

    app = ApplicationBuilder().token(bot_token).build()
    listener = MessageListener(allowed_chat_ids)

    app.add_handler(MessageHandler(filters.ALL, listener.on_message))
    app.run_polling()


if __name__ == "__main__":
    main()
