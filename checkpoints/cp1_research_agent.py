#!/usr/bin/env python3
# CHECKPOINT 1 build - Prompt 1 "the program".
# This is the first stage of the research agent. At this point the program
# can only do three things:
#   1. Read the model settings (API_BASE_URL, API_KEY, MODEL) from .env.
#   2. Ask the user for a research question and print it back.
#   3. Send that question to the model and print whatever the model answers.
#
# There are no tools yet. The agent does not search or read pages yet - that
# comes in checkpoint 2. This stage exists so we can check that the
# connection to the model works before building anything on top of it.

import os
import sys
import time

import requests

# Many websites and the model API refuse requests that identify themselves as
# a script. Sending a browser-style User-Agent gets past that. Without it the
# service returns 403 with error code 1010.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def load_env():
    # Read the three settings from .env so they live outside the code. That
    # way, changing the model later is editing a settings file, not the
    # program. Stop with a clear message if any setting is missing, rather
    # than failing later in a confusing place.
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    values = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    missing = [k for k in ("API_BASE_URL", "API_KEY", "MODEL") if not values.get(k)]
    if missing:
        print("Missing setting(s) in .env: " + ", ".join(missing))
        sys.exit(1)
    return values


def call_model(messages):
    # Send the conversation to the model over plain HTTP and return the reply
    # text. The key goes in an Authorization header, never in the URL - a key
    # in a URL ends up in every error message and is how keys get published by
    # accident.
    #
    # Retries: 429 means "ask again later" (wait as long as Retry-After
    # asks), 5xx means the service hiccuped (wait two seconds and retry).
    # 401/403 usually means the key is wrong, so print a hint and stop
    # instead of retrying.
    url = API_BASE_URL + "/chat/completions"
    headers = {
        "Authorization": "Bearer " + API_KEY,
        "User-Agent": BROWSER_USER_AGENT,
        "Content-Type": "application/json",
    }
    body = {"model": MODEL, "messages": messages}
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=60)
        except requests.RequestException as e:
            print(f"Request failed: {type(e).__name__}: {e}")
            time.sleep(2)
            continue
        if resp.status_code == 200:
            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content")
            if not content:
                print("Reply is missing choices[0].message.content. Full response body:")
                print(resp.text)
                sys.exit(1)
            return content
        if resp.status_code == 429:
            wait = resp.headers.get("Retry-After")
            wait = int(wait) if wait and wait.isdigit() else 2
            print(f"Rate limited (429). Waiting {wait} seconds, then retrying.")
            time.sleep(wait)
            continue
        if 500 <= resp.status_code < 600:
            print(f"Server error {resp.status_code}. Waiting 2 seconds, then retrying.")
            time.sleep(2)
            continue
        print(f"Error {resp.status_code}: {resp.text}")
        if resp.status_code in (401, 403):
            print("This looks like an authentication or access problem. Check API_KEY in your .env file.")
            sys.exit(1)
    print("Gave up after retrying 3 times.")
    sys.exit(1)


def main():
    global API_BASE_URL, API_KEY, MODEL
    env = load_env()
    API_BASE_URL = env["API_BASE_URL"]
    API_KEY = env["API_KEY"]
    MODEL = env["MODEL"]

    question = input("Research question: ")
    print("Question: " + question)

    reply = call_model([
        {"role": "user", "content": question},
    ])
    print()
    print(reply)


if __name__ == "__main__":
    main()