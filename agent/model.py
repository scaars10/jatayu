from __future__ import annotations

from agent.gemini_model import gemini_model, get_client
from config.env_config import init_config


def main() -> None:
    init_config()
    client = get_client()
    response = client.models.generate_content(
        contents="Write a short story about a robot learning to love:",
        model=gemini_model.get_light_model(),
    )
    print(response.text)


if __name__ == "__main__":
    main()
