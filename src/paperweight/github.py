"""Small GitHub REST client for creating the output repository and seeding it."""

from __future__ import annotations

import base64
from typing import Any

import httpx


class GitHubError(RuntimeError):
    pass


class GitHubClient:
    def __init__(
        self,
        token: str,
        base_url: str = "https://api.github.com",
        client: httpx.Client | None = None,
    ):
        self.http = client or httpx.Client(timeout=httpx.Timeout(60.0))
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        response = self.http.request(
            method, f"{self.base_url}{path}", headers=self.headers, **kwargs
        )
        if response.status_code >= 400 and response.status_code != 404:
            raise GitHubError(
                f"GitHub {method} {path} failed ({response.status_code}): {response.text[:500]}"
            )
        return response

    def authenticated_login(self) -> str:
        response = self._request("GET", "/user")
        if response.status_code == 404:
            raise GitHubError("GitHub token cannot read /user")
        return response.json()["login"]

    def get_repo(self, owner: str, name: str) -> dict[str, Any] | None:
        response = self._request("GET", f"/repos/{owner}/{name}")
        return None if response.status_code == 404 else response.json()

    def ensure_repo(
        self, owner: str, name: str, *, private: bool = True, description: str = ""
    ) -> dict[str, Any]:
        """Return the repository, creating it (with an initial commit) if it does not exist."""
        existing = self.get_repo(owner, name)
        if existing:
            return existing
        payload = {
            "name": name,
            "private": private,
            "description": description[:350],
            "auto_init": True,
        }
        if owner.lower() == self.authenticated_login().lower():
            response = self._request("POST", "/user/repos", json=payload)
        else:
            response = self._request("POST", f"/orgs/{owner}/repos", json=payload)
        if response.status_code == 404:
            raise GitHubError(f"cannot create repository under {owner!r}")
        return response.json()

    def put_file(
        self, owner: str, name: str, path: str, content: str, message: str, branch: str
    ) -> None:
        existing = self._request(
            "GET", f"/repos/{owner}/{name}/contents/{path}", params={"ref": branch}
        )
        payload: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch,
        }
        if existing.status_code == 200 and isinstance(existing.json(), dict):
            payload["sha"] = existing.json()["sha"]
        response = self._request("PUT", f"/repos/{owner}/{name}/contents/{path}", json=payload)
        if response.status_code == 404:
            raise GitHubError(f"cannot write {path} to {owner}/{name}@{branch}")

    def list_pulls(self, owner: str, name: str, state: str = "open") -> list[dict[str, Any]]:
        response = self._request("GET", f"/repos/{owner}/{name}/pulls", params={"state": state})
        return [] if response.status_code == 404 else response.json()
