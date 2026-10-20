"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --interactive  # 対話モード（/exit で終了）

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)

    print_usage(response)
    return response.text


def print_usage(response) -> None:
    usage = response.usage_metadata
    if usage is not None:
        print(f"[入力トークン数] {usage.prompt_token_count}")
        print(f"[出力トークン数] {usage.candidates_token_count}")


def create_chat():
    client = genai.Client(api_key=get_api_key())
    return client, client.chats.create(model=MODEL)


def ask_in_chat(question: str, chat) -> str:
    response = chat.send_message(question)
    print_usage(response)
    return response.text


def interactive_chat(first_question: str | None = None) -> None:
    client, chat = create_chat()
    question = first_question
    print("対話モードです。終了するには /exit と入力してください。")
    while True:
        if question is None:
            try:
                question = input("あなた> ").strip()
            except EOFError:
                break
        if question.lower() in {"/exit", "/quit"}:
            break
        if question:
            print(ask_in_chat(question, chat))
        question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--interactive", action="store_true", help="会話履歴を保つ対話モード")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.interactive:
        try:
            interactive_chat(args.question)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            raise SystemExit(1)
        return

    question = args.question or DEFAULT_QUESTION
    print(f"[質問] {question}\n")
    try:
        print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
