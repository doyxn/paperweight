from __future__ import annotations

import json

import httpx
import pytest
import respx

from paperweight.devin import DevinClient, DevinError, devin_id, is_settled
from paperweight.github import GitHubClient

DEVIN = "https://api.devin.ai/v3/organizations/org-1"
GH = "https://api.github.com"


def test_devin_id_prefix():
    assert devin_id("abc") == "devin-abc"
    assert devin_id("devin-abc") == "devin-abc"


@pytest.mark.parametrize(
    ("status", "detail", "settled"),
    [
        ("running", "working", False),
        ("running", "waiting_for_approval", False),
        ("running", "finished", True),
        ("running", "waiting_for_user", True),
        ("exit", None, True),
        ("suspended", "out_of_credits", True),
        ("new", None, False),
    ],
)
def test_is_settled(status, detail, settled):
    assert is_settled({"status": status, "status_detail": detail}) is settled


@respx.mock
def test_devin_create_and_wait():
    create = respx.post(f"{DEVIN}/sessions").mock(
        return_value=httpx.Response(200, json={"session_id": "devin-1", "url": "u"})
    )
    respx.get(f"{DEVIN}/sessions/devin-1").mock(
        side_effect=[
            httpx.Response(200, json={"status": "running", "status_detail": "working"}),
            httpx.Response(200, json={"status": "running", "status_detail": "working"}),
            httpx.Response(200, json={"status": "running", "status_detail": "finished"}),
        ]
    )
    sleeps: list[float] = []
    client = DevinClient("key", "org-1", sleep=sleeps.append)
    session = client.create_session(
        "do it", title="t", session_secrets={"GITHUB_TOKEN": "tok"}, max_acu_limit=5
    )
    body = json.loads(create.calls.last.request.content)
    assert create.calls.last.request.headers["Authorization"] == "Bearer key"
    assert body["session_secrets"] == [{"key": "GITHUB_TOKEN", "value": "tok", "sensitive": True}]
    assert body["max_acu_limit"] == 5

    updates: list[dict] = []
    final = client.wait(session["session_id"], poll_interval=1, on_update=updates.append)
    assert final["status_detail"] == "finished"
    assert sleeps == [1, 1]
    assert len(updates) == 2


@respx.mock
def test_devin_error_surfaces():
    respx.post(f"{DEVIN}/sessions").mock(return_value=httpx.Response(403, text="forbidden"))
    with pytest.raises(DevinError, match="403"):
        DevinClient("key", "org-1").create_session("x")


@respx.mock
def test_github_ensure_repo_creates_under_user():
    respx.get(f"{GH}/repos/me/new").mock(return_value=httpx.Response(404))
    respx.get(f"{GH}/user").mock(return_value=httpx.Response(200, json={"login": "Me"}))
    create = respx.post(f"{GH}/user/repos").mock(
        return_value=httpx.Response(201, json={"default_branch": "main"})
    )
    repo = GitHubClient("tok").ensure_repo("me", "new", private=True, description="d")
    assert repo["default_branch"] == "main"
    assert json.loads(create.calls.last.request.content)["auto_init"] is True


@respx.mock
def test_github_ensure_repo_creates_under_org():
    respx.get(f"{GH}/repos/acme/new").mock(return_value=httpx.Response(404))
    respx.get(f"{GH}/user").mock(return_value=httpx.Response(200, json={"login": "me"}))
    create = respx.post(f"{GH}/orgs/acme/repos").mock(
        return_value=httpx.Response(201, json={"default_branch": "main"})
    )
    GitHubClient("tok").ensure_repo("acme", "new")
    assert create.called


@respx.mock
def test_github_put_file_updates_existing():
    path = f"{GH}/repos/me/r/contents/docs/a.md"
    respx.get(path).mock(return_value=httpx.Response(200, json={"sha": "abc"}))
    put = respx.put(path).mock(return_value=httpx.Response(200, json={}))
    GitHubClient("tok").put_file("me", "r", "docs/a.md", "hi", "msg", "main")
    body = json.loads(put.calls.last.request.content)
    assert body["sha"] == "abc" and body["branch"] == "main"
