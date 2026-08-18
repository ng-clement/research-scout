#!/usr/bin/env python3
# CHECKPOINT 3 build - Prompt 3 "the loop".
# This stage turns the program into a tool-using agent with an explicit loop.
# The loop (called ReAct - reason, act, observe) works like this:
#   1. Send the goal, the tool descriptions, and everything that has happened
#      so far to the model.
#   2. The model replies with ONLY a JSON object saying which tool to use:
#      SEARCH, READ or FINISH.
#   3. The program runs the tool and appends the result to a state list.
#   4. Repeat, sending the WHOLE state each time, because the model has no
#      memory of its own - the state list IS the run's only memory.
#
# The step limit is a constant at the top of the file, and the same number is
# put into the prompt text sent to the model, so what the model is told and
# what the program enforces can never drift apart.

import json
import os
import re
import sys
import time

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

# Maximum number of times the loop asks the model before giving up.
STEP_LIMIT = 6

# A web page can be enormous. Cap how much text read_webpage returns so one
# page cannot fill the whole request and crowd everything else out.
MAX_PAGE_CHARS = 2000

# A browser-style User-Agent. Without one the service refuses the request
# with a 403 and error code 1010.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# The instructions the model is given at the start of every run. It names the
# three tools, forces the reply into one of three JSON shapes (the contract
# between the two halves of the program), and fixes the report's structure.
# The f-string below fills in MAX_PAGE_CHARS and STEP_LIMIT so the numbers in
# the prompt and in the code always match.
SYSTEM_PROMPT = f"""You are a research agent. You answer a research question by searching the web, reading pages, and finally writing a short research brief.

You have three tools:
SEARCH  Search the web for a query and get up to 5 results, each with a title, a URL and a snippet. Use this when you need to find sources.
READ    Open one web page and return up to {MAX_PAGE_CHARS} characters of visible text. Use this to read a page before you rely on it.
FINISH  Write the report. Only choose this when you have enough information to answer the question.

The report must have four sections:
Findings:
  one or more findings
Comparison:
  compare the options or evidence you found
Recommendation:
  your recommendation based on what you read
Sources:
  the URLs you used, one per line

Reply with ONLY a JSON object, in one of these three shapes:
{{"reason": "one short sentence", "action": "SEARCH", "query": "..."}}
{{"reason": "one short sentence", "action": "READ", "url": "..."}}
{{"reason": "one short sentence", "action": "FINISH", "report": "..."}}

No other text. The run is limited to {STEP_LIMIT} steps."""


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
    # Send the accumulated conversation to the model and get its reply back.
    # The key goes in an Authorization header, never in the URL. Handles the
    # three kinds of failure: 429 (rate limited - wait), 5xx (server hiccup -
    # wait), 401/403 (bad key - stop with a hint).
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
    # Tool 1 of 3. Uses ddgs, the DuckDuckGo search library, which needs no
    # API key. On failure it returns an empty list so the loop can keep going
    # instead of crashing the run.
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
    # Tool 2 of 3. Fetch one page, strip the HTML tags, and return up to
    # MAX_PAGE_CHARS characters of visible text. Web page text is often full
    # of navigation and menus, so do not assume it is clean. Pages that
    # refuse to load are normal (many commercial sites block scripts); return
    # "" and let the agent choose somewhere else.
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
    return text[:MAX_PAGE_CHARS]


def parse_reply(text):
    # The model's reply must be a JSON object in one of three shapes. Models
    # often wrap their reply in ``` markdown fences, so strip those first. If
    # it still cannot be parsed, print the raw reply and stop - a
    # wrong-shaped reply means the loop cannot continue.
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    print("Could not parse the model's reply as JSON. Raw reply:")
    print(text)
    sys.exit(1)


def validate_reply(parsed):
    # Check the parsed JSON has the field the claimed action needs (a query
    # for SEARCH, a URL for READ, a report for FINISH). Keeps a malformed
    # reply from crashing the loop later.
    action = parsed.get("action")
    if action == "SEARCH":
        return "query" in parsed
    if action == "READ":
        return "url" in parsed
    if action == "FINISH":
        return "report" in parsed
    return False


def format_state(state):
    # Turn the state list into text the model can read. Every step becomes one
    # line: the step number, the action, a short summary, and what came back
    # (the observation). This whole text is sent to the model each step.
    if not state:
        return "(nothing yet)"
    return "\n".join(
        f"Step {i}: {entry.get('action')} | {entry.get('summary')} | observation: {entry.get('observation')}"
        for i, entry in enumerate(state, start=1)
    )


def split_report(report):
    # Break the report's text into its sections by finding the heading lines.
    # Returns a flag saying whether all four were found, so the brief can fall
    # back to printing the raw report if the model skipped a heading.
    sections = {"Findings": "", "Comparison": "", "Recommendation": "", "Sources": ""}
    current = None
    for line in report.splitlines():
        m = re.match(r"^\s*(Findings|Comparison|Recommendation|Sources)\s*:", line)
        if m:
            current = m.group(1)
            continue
        if current:
            sections[current] += line + "\n"
    for key in sections:
        sections[key] = sections[key].strip()
    if all(sections.values()):
        return sections, True
    return sections, False


def print_brief(question, report):
    # Print the finished brief: the question, the report sections, and the
    # sources. This is the program-side version of "FINISH".
    sections, ok = split_report(report)
    print("\nResearch brief")
    print("=============")
    print("Question:")
    print("  " + question)
    print()
    if ok:
        print("Findings:")
        print(sections["Findings"])
        print()
        print("Comparison:")
        print(sections["Comparison"])
        print()
        print("Recommendation:")
        print(sections["Recommendation"])
        print()
        print("Sources:")
        print(sections["Sources"])
    else:
        print("Findings and analysis:")
        print(report.strip())


def run_agent(question):
    # The heart of the agent: the ReAct loop. Each step:
    #   1. Send the goal, the tool descriptions, and the whole state to the
    #      model, asking for a JSON tool call.
    #   2. Read its reply and do what it asked (SEARCH / READ / FINISH).
    #   3. Append the result to state so the next step can see it.
    # The model has no memory of its own, so state IS the run's only memory.
    #
    # A failed search, or a page that will not load, is normal. Record it in
    # the state so the model can see it on the next step, and carry on. The
    # program does NOT stop on the first failure.
    state = []
    finished = False
    report = None
    steps_used = 0

    for step in range(1, STEP_LIMIT + 1):
        steps_used = step
        user_message = (
            f"Research question: {question}\n\n"
            f"This is step {step} of up to {STEP_LIMIT}.\n\n"
            f"What has happened so far:\n{format_state(state)}\n\n"
            "What do you do next? Reply with only the JSON object."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        reply = call_model(messages)
        parsed = parse_reply(reply)
        if not validate_reply(parsed):
            print("Model reply was not a valid tool call. Raw reply:")
            print(reply)
            sys.exit(1)

        reason = parsed.get("reason", "")
        action = parsed.get("action", "")

        if action == "SEARCH":
            query = parsed.get("query", "")
            results = search_web(query)
            summary = f"{len(results)} results"
            print(f"STEP {step}  SEARCH  {query!r}  ->  {summary}  (reason: {reason})")
            listing = "; ".join(f"{r['title']} {r['url']}" for r in results)
            state.append({
                "action": "SEARCH",
                "summary": f"SEARCH {query!r}",
                "observation": f"{summary}. {listing}",
            })

        elif action == "READ":
            url = parsed.get("url", "")
            text = read_webpage(url)
            print(f"STEP {step}  READ  {url}  ->  {len(text)} characters  (reason: {reason})")
            state.append({
                "action": "READ",
                "summary": f"READ {url}",
                "observation": f"Got {len(text)} characters from {url}. Text starts: {text[:300]}",
            })

        elif action == "FINISH":
            report = parsed.get("report", "")
            if not report.strip():
                print(f"STEP {step}  FINISH  ->  refused: report is empty  (reason: {reason})")
                state.append({
                    "action": "FINISH",
                    "summary": "FINISH",
                    "observation": "REFUSED: the report was empty. Write a real report, or choose READ.",
                })
                continue
            finished = True
            print(f"STEP {step}  FINISH  (reason: {reason})")
            print_brief(question, report)
            break

        else:
            print(f"STEP {step}  unknown action {action!r}  (reason: {reason})")
            state.append({
                "action": action,
                "summary": action,
                "observation": "Unknown action. Choose SEARCH, READ or FINISH.",
            })

    if not finished:
        print("\nStep limit reached without finishing. The agent used all " + str(STEP_LIMIT) + " steps.")

    return state, report, finished, steps_used


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

    run_agent(question)


if __name__ == "__main__":
    main()