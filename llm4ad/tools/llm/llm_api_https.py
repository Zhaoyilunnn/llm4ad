# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import httpx
import time
from typing import Any, Dict, Optional  # Added Dict, Optional
import traceback
from ...base import LLM


class HttpsApi(LLM):
    def __init__(
        self,
        host: str,
        key: str,
        model: str,
        timeout: int = 60,
        **kwargs,
    ):
        """Https API
        Args:
            host   : host name. please note that the host name does not include 'https://'
            key    : API key.
            model  : LLM model name.
            timeout: API timeout.
            proxies: Optional dictionary of proxies to use (e.g., {"http://": "http://localhost:8030", "https://": "socks5://localhost:1080"}).
                     If None, httpx will try to use environment variables (HTTP_PROXY, HTTPS_PROXY, ALL_PROXY).
        """
        super().__init__(**kwargs)
        self._host = host
        self._key = key
        self._model = model
        self._timeout = timeout
        self._kwargs = kwargs
        self._cumulative_error = 0

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt.strip()}]

        url = f"https://{self._host}/v1/chat/completions"
        json_payload = {
            "max_tokens": self._kwargs.get("max_tokens", 4096),
            "top_p": self._kwargs.get("top_p", None),
            "temperature": self._kwargs.get("temperature", 1.0),
            "model": self._model,
            "messages": prompt,
        }
        headers = {
            "Authorization": f"Bearer {self._key}",
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
        }

        while True:
            try:
                # httpx.Client by default respects environment variables for proxies (trust_env=True)
                # if self._proxies is None.
                with httpx.Client(timeout=self._timeout) as client:
                    res = client.post(url, json=json_payload, headers=headers)
                    res.raise_for_status()  # Raise an exception for HTTP 4xx or 5xx status codes
                    data = res.json()

                response = data["choices"][0]["message"]["content"]
                if (
                    self.debug_mode
                ):  # Assuming self.debug_mode is defined in the base LLM class
                    self._cumulative_error = 0
                return response
            except httpx.HTTPStatusError as e:
                self._cumulative_error += 1
                error_message = f"{self.__class__.__name__} HTTP error: {e.response.status_code} - {e.response.text}"
                if self.debug_mode:
                    if self._cumulative_error == 10:
                        raise RuntimeError(
                            f"{error_message}. You may check your API host, API key, or proxy configuration."
                        )
                else:
                    print(error_message)
                    print(
                        f"You may check your API host, API key, or proxy configuration."
                    )
                    time.sleep(2)
                continue
            except (
                httpx.RequestError
            ) as e:  # Catches other httpx request errors (network, timeout, etc.)
                self._cumulative_error += 1
                error_message = f"{self.__class__.__name__} request error ({type(e).__name__}): {traceback.format_exc()}."
                if self.debug_mode:
                    if self._cumulative_error == 10:
                        raise RuntimeError(
                            f"{error_message} You may check your API host, API key, or proxy configuration."
                        )
                else:
                    print(error_message)
                    print(
                        f"You may check your API host, API key, or proxy configuration."
                    )
                    time.sleep(2)
                continue
            except Exception as e:  # General fallback for unexpected errors
                self._cumulative_error += 1
                error_message = f"{self.__class__.__name__} unexpected error: {traceback.format_exc()}."
                if self.debug_mode:
                    if self._cumulative_error == 10:
                        raise RuntimeError(
                            f"{error_message} You may check your API host, API key, or proxy configuration."
                        )
                else:
                    print(error_message)
                    print(
                        f"You may check your API host, API key, or proxy configuration."
                    )
                    time.sleep(2)
                continue
