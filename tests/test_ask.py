from types import SimpleNamespace

from app import ask


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=7,
        ),
        text="answer",
    )

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                return response

    monkeypatch.setattr(ask.genai, "Client", lambda api_key: FakeClient())
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    assert ask.ask("question") == "answer"
    assert capsys.readouterr().out == "[入力トークン数] 12\n[出力トークン数] 7\n"


def test_interactive_chat_reuses_one_chat_for_all_turns(monkeypatch, capsys):
    class FakeChat:
        def __init__(self):
            self.questions = []

        def send_message(self, question):
            self.questions.append(question)
            return SimpleNamespace(usage_metadata=None, text=f"reply to {question}")

    chat = FakeChat()
    answers = iter(["first question", "follow-up", "/exit"])
    monkeypatch.setattr(ask, "create_chat", lambda: (object(), chat))
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    ask.interactive_chat()

    assert chat.questions == ["first question", "follow-up"]
    assert "reply to follow-up" in capsys.readouterr().out