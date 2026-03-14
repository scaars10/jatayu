from __future__ import annotations

from dotenv import load_dotenv


from agent.gemini_model import gemini_model, get_client


def main() -> None:
    load_dotenv()
    client = get_client()
    response = client.models.generate_content(
        contents="Write a short story about a robot learning to love:",
        model=gemini_model.get_light_model(),
    )
    print(response.text)


if __name__ == "__main__":
    main()
