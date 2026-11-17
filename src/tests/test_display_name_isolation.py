import re
from pathlib import Path

import pytest
from agent_helpers import chat_json, make_client

from getwork.webapp import demo

pytestmark = pytest.mark.usefixtures("offline")

NAME_PATTERN = re.compile(r"pseudonym|display_name", re.IGNORECASE)
ALLOWED = {"webapp/pseudonym.py", "webapp/api.py", "webapp/models.py"}
CANDIDATE_REQUEST = {
    "direction": "posting_to_resume",
    "query": {"kind": "corpus", "id": "j-1"},
    "k": 6,
}


def test_only_allowed_modules_mention_names() -> None:
    root = Path(__file__).resolve().parents[1] / "getwork"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if relative in ALLOWED or "__pycache__" in path.parts:
            continue
        if NAME_PATTERN.search(path.read_text(encoding="utf-8")):
            offenders.append(relative)
    assert offenders == [], f"modules mention pseudonym/display_name: {offenders}"


def test_chat_prompts_contain_no_names() -> None:
    client, fake = make_client(["Both documents list JavaScript and CSS."])
    matched = client.post("/api/match", json=CANDIDATE_REQUEST)
    assert matched.status_code == 200
    names = [match["display_name"] for match in matched.json()["matches"]]
    assert names and all(names)

    body = chat_json(
        direction="posting_to_resume",
        query=demo.DemoEngine().get_document("j-1").model_dump(mode="json"),
        focus_id="r-1",
        matches=[{"id": "r-1", "rank": 1, "score": 0.5}, {"id": "r-2", "rank": 2, "score": 0.4}],
        message="Why is r-1 ranked above r-2?",
    )
    reply = client.post("/api/chat", json=body)
    assert reply.status_code == 200
    assert fake.calls
    seen = [call.system + "\n" + call.user for call in fake.calls] + [reply.text]
    for name in names:
        assert not any(name in text for text in seen), name


def test_match_request_writes_no_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert list(tmp_path.iterdir()) == []
    client, _ = make_client([])
    response = client.post("/api/match", json=CANDIDATE_REQUEST)
    assert response.status_code == 200
    assert list(tmp_path.iterdir()) == []
