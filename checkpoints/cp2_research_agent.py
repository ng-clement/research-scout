#!/usr/bin/env python3
# CHECKPOINT 2 build - Prompt 2 "the tools".
# This stage adds the two functions that let the agent touch the outside
# world. They are the ONLY two tools the agent has:
#   search_web(query)   - find web pages about a topic
#   read_webpage(url)   - open one page and return its visible text
#
# Anything not in this list, the agent cannot do. The one-sentence
# descriptions ("docstrings") are what the model actually sees when it
# decides which tool to call in a later checkpoint. Each tool can also be
# tested on its own from the command line:
#   python research_agent.py --search "my query"
#   python research_agent.py --read https://example.com

import os
import sys
import time

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

# A browser-style User-Agent. Without one the service refuses the request
# with a 403 and error code 1010, because it sees a script instead of a
# browser.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def load_env():
    # Read the three settings from .env so they live outside the code. Stop
    # with a clear message if any setting is missing.
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
    # text. The key goes in an Authorization header, never in the URL.
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


def search_web(query):
    """Search the web for a query and return up to 5 results, each with a title, a URL and a snippet."""
    # Tool 1 of 2. Uses ddgs, the DuckDuckGo search library, which needs no
    # API key. On failure it returns an empty list so the program can keep
    # going instead of stopping.
    results = []
    try:
        raw = DDGS().text(query, max_results=5, timeout=20)
        for item in raw:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("href") or item.get("url", ""),
                "snippet": item.get("body", ""),
            })
    except Exception as e:
        print(f"search_web failed for query {query!r}: {type(e).__name__}: {e}")
        return []
    return results


def read_webpage(url):
    """Fetch one web page and return up to 2000 characters of its visible text, or an empty string on failure."""
    # Tool 2 of 2. Fetch one page, strip the HTML tags, and return up to 2000
    # characters of visible text. Pages that refuse to load are normal (many
    # commercial sites block scripts); return "" and let the caller choose
    # somewhere else.
    try:
        resp = requests.get(url, headers={"User-Agent": BROWSER_USER_AGENT}, timeout=25)
    except requests.RequestException as e:
        print(f"read_webpage failed for {url}: {type(e).__name__}: {e}")
        return ""
    if resp.status_code != 200:
        print(f"read_webpage: status {resp.status_code} for {url}")
        return ""
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    text = " ".join(text.split())
    return text[:2000]


def main():
    # Command-line entry point.
    #   python research_agent.py                  normal run (asks for a question)
    #   python research_agent.py --search "<query>"   test the search tool alone
    #   python research_agent.py --read <url>         test the page reader alone
    args = sys.argv[1:]
    if args and args[0] == "--search":
        query = " ".join(args[1:])
        for r in search_web(query):
            print(r["title"])
            print(r["url"])
            print(r["snippet"])
            print()
        return
    if args and args[0] == "--read":
        url = args[1]
        print(read_webpage(url))
        return

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