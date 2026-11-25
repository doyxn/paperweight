"""Devin API v3 client: create sessions, poll them, and collect their results."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

TERMINAL_STATUSES = {"exit", "error", "suspended"}
TERMINAL_DETAILS = {"finished", "waiting_for_user"}


class DevinError(RuntimeError):
    pass


def devin_id(session_id: str) -> str:
    return session_id if session_id.startswith("devin-") else f"devin-{session_id}"


def is_settled(session: dict[str, Any]) -> bool:
    return (
        session.get("status") in TERMINAL_STATUSES
        or session.get("status_detail") in TERMINAL_DETAILS
    )


class DevinClient:
    def __init__(
        self,
        api_key: str,
        org_id: str,
        base_url: str = "https://api.devin.ai",
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.http = client or httpx.Client(timeout=httpx.Timeout(60.0))
        self.base_url = base_url.rstrip("/")
        self.org_id = org_id
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self._sleep = sleep

    def _url(self, path: str) -> str:
        return f"{self.base_url}/v3/organizations/{self.org_id}{path}"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self.http.request(method, self._url(path), headers=self.headers, **kwargs)
        if response.status_code >= 400:
            raise DevinError(
                f"Devin {method} {path} failed ({response.status_code}): {response.text[:500]}"
            )
        return response.json()

    def create_session(
        self,
        prompt: str,
        *,
        title: str | None = None,
        tags: list[str] | None = None,
        structured_output_schema: dict[str, Any] | None = None,
        session_secrets: dict[str, str] | None = None,
        repos: list[str] | None = None,
        max_acu_limit: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"prompt": prompt}
        if title:
            body["title"] = title
        if tags:
            body["tags"] = tags
        if structured_output_schema:
            body["structured_output_schema"] = structured_output_schema
        if session_secrets:
            body["session_secrets"] = [
                {"key": k, "value": v, "sensitive": True} for k, v in session_secrets.items()
            ]
        if repos:
            body["repos"] = repos
        if max_acu_limit:
            body["max_acu_limit"] = max_acu_limit
        return self._request("POST", "/sessions", json=body)

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/sessions/{devin_id(session_id)}")

    def send_message(self, session_id: str, message: str) -> dict[str, Any]:
        return self._request(
            "POST", f"/sessions/{devin_id(session_id)}/messages", json={"message": message}
        )

    def wait(
        self,
        session_id: str,
        *,
        poll_interval: float = 30.0,
        timeout: float = 4 * 3600,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Poll until the session finishes, blocks on the user, or stops."""
        deadline = time.monotonic() + timeout
        last_state: tuple[Any, Any] | None = None
        while True:
            session = self.get_session(session_id)
            state = (session.get("status"), session.get("status_detail"))
            if on_update and state != last_state:
                on_update(session)
            last_state = state
            if is_settled(session):
                return session
            if time.monotonic() >= deadline:
                raise DevinError(f"timed out waiting for session {session_id} (last: {state})")
            self._sleep(poll_interval)
