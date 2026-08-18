#!/usr/bin/env python3
# A research agent that works in a loop: ask the model what to do, act with a
# tool, look at the result, and ask again. This shape is called ReAct
# (reason + act) and comes from Yao et al. 2022, arxiv.org/abs/2210.03629.
#
# Seven parts make up the agent:
#   Model       - the LLM that decides what to do next (call_model)
#   Goal        - the research question typed by the user
#   Tools       - search_web and read_webpage; anything not listed the agent
#                 cannot do
#   State       - the list of every step so far. The model has no memory of
#                 its own, so the whole list is sent back to it every step
#   Loop        - run_agent asks, acts, observes, and repeats
#   Stop        - the step limit and the "must read three pages" rule
#   Evaluation  - the five checks in evaluate(), run with --eval
import json
import os
import re
import sys
import time

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

# Maximum number of times the loop asks the model before giving up. The model
# is told this same number so what it is told and what the program enforces
# can never drift apart.
STEP_LIMIT = 8

# A web page can be enormous. Cap how much text read_webpage returns so one
# page cannot fill the whole request and crowd everything else out.
MAX_PAGE_CHARS = 5000

# The agent may only finish after it has read this many DIFFERENT pages that
# actually returned text. Search titles and snippets are not enough.
MIN_PAGES_TO_READ = 3

# Many websites and the model API refuse requests that identify themselves as
# a script. Sending a browser-style User-Agent gets past that.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# The description of the FINISH tool that the model sees. The model has never
# seen this program's code, so this sentence IS the tool as far as it is
# concerned. Important: this is only a REQUEST to the model. The matching rule
# in run_agent (refusing FINISH before three pages) is what the model cannot
# ignore.
FINISH_DESCRIPTION = (
    "FINISH  Write the report. Only choose this after you have read at least three "
    "different web pages. Base the report on the text of those pages. Search result "
    "titles and snippets are not enough on their own. Price tickers, shop pages and "
    "product listings give you a number but no explanation, so prefer news articles, "
    "analysis and official sources when you choose what to read."
)

SYSTEM_PROMPT = f"""You are a research agent. You answer a research question by searching the web, reading pages, and finally writing a short research brief.

You have three tools:
SEARCH  Search the web for a query and get up to 5 results, each with a title, a URL and a snippet. Use this when you need to find sources.
READ    Open one web page and return up to {MAX_PAGE_CHARS} characters of visible text. Use this to read a page before you rely on it. Do not read the same URL twice.
{FINISH_DESCRIPTION}

The report must have three sections, each on its own heading line:
Findings:
  one or more findings, each ending with the URL it came from in square brackets, like [https://example.com]
Comparison:
  compare the options or evidence you found
Recommendation:
  your recommendation based on what you read

Reply with ONLY a JSON object, in one of these three shapes:
{{"reason": "one short sentence", "action": "SEARCH", "query": "..."}}
{{"reason": "one short sentence", "action": "READ", "url": "..."}}
{{"reason": "one short sentence", "action": "FINISH", "report": "..."}}

No other text. The run is limited to {STEP_LIMIT} steps. Only choose FINISH after you have read at least three different web pages."""

# The instructions the model is given at the start of every run. It names the
# three tools, forces the reply into one of three JSON shapes (the contract
# between the two halves of the program), and fixes the report's structure.


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
    # Send the accumulated conversation to the model and get its reply back.
    # The key goes in an Authorization header, never in the URL - a key in a
    # URL ends up in every error message and is how keys get published by
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
    """Fetch one web page and return up to 5000 characters of its visible text, or an empty string on failure."""
    # Tool 2 of 3. Fetch one page, strip the HTML tags, and return up to
    # MAX_PAGE_CHARS characters of visible text. Pages that refuse to load
    # are normal (many commercial sites block scripts); return "" and let the
    # agent choose somewhere else. A page that fails here is NOT a source.
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
    # Break the report's text into the three required sections by finding the
    # "Findings:", "Comparison:" and "Recommendation:" heading lines. Returns
    # a flag saying whether all three were found, so the brief can fall back
    # to printing the raw report if the model skipped a heading.
    sections = {"Findings": "", "Comparison": "", "Recommendation": ""}
    current = None
    for line in report.splitlines():
        m = re.match(r"^\s*(Findings|Comparison|Recommendation)\s*:", line)
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


def print_brief(question, report, pages_read, also_found):
    # Print the finished brief: the question, the three report sections, and
    # an honest source list. Two lists, deliberately separate: "Pages read"
    # (pages that actually returned text) and "Also found, not opened" (URLs
    # that only appeared in search results). A page that failed to load is in
    # neither list.
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
    else:
        print("Findings and analysis:")
        print(report.strip())
    print()
    print("Pages read:")
    seen = set()
    for url in pages_read:
        if url not in seen:
            print("  " + url)
            seen.add(url)
    print("Also found, not opened:")
    seen2 = set()
    for url in also_found:
        if url not in seen2:
            print("  " + url)
            seen2.add(url)


def run_agent(question):
    # The heart of the agent: the ReAct loop. Each step:
    #   1. Send the goal, the tool descriptions, and the whole state to the
    #      model, asking for a JSON tool call.
    #   2. Read its reply and do what it asked (SEARCH / READ / FINISH).
    #   3. Append the result to state so the next step can see it.
    # The model has no memory of its own, so state IS the run's only memory.
    #
    # The stop conditions live HERE, in the program, not in a request to the
    # model: the step limit, the three-page rule, refusing empty reports, and
    # refusing to read a URL twice. Those are rules the model cannot talk its
    # way past.
    state = []
    pages_read = []
    also_found = []
    tried_urls = set()
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
            # Anything the search returns that has not been seen before is
            # added to "also found" for now. It moves to "pages read" only if
            # the agent later opens it and gets text back.
            for r in results:
                url = r["url"]
                if url not in pages_read and url not in also_found:
                    also_found.append(url)
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
            # Program-side rule: never read the same URL twice. If the model
            # asks for one it already read, refuse and put that in state.
            if url in tried_urls:
                print(f"STEP {step}  READ  {url}  ->  refused (already read)  (reason: {reason})")
                state.append({
                    "action": "READ",
                    "summary": f"READ {url}",
                    "observation": "REFUSED: this URL was already read. Choose a different URL.",
                })
                continue
            tried_urls.add(url)
            text = read_webpage(url)
            if text:
                # Only count it as a source when the page actually returned
                # text. A page that failed to load is neither read nor a
                # source.
                pages_read.append(url)
                also_found = [u for u in also_found if u != url]
                print(f"STEP {step}  READ  {url}  ->  {len(text)} characters  (reason: {reason})")
                state.append({
                    "action": "READ",
                    "summary": f"READ {url}",
                    "observation": f"Got {len(text)} characters from {url}. Text starts: {text[:300]}",
                })
            else:
                print(f"STEP {step}  READ  {url}  ->  0 characters (refused)  (reason: {reason})")
                state.append({
                    "action": "READ",
                    "summary": f"READ {url}",
                    "observation": f"Could not read {url}; got no text. Choose another URL.",
                })

        elif action == "FINISH":
            report = parsed.get("report", "")
            # Program-side rules the model cannot bypass. Refuse an empty
            # report, and refuse FINISH before the agent has actually read
            # three different pages (a description in the prompt is only a
            # request; this rule is not).
            if not report.strip():
                print(f"STEP {step}  FINISH  ->  refused: report is empty  (reason: {reason})")
                state.append({
                    "action": "FINISH",
                    "summary": "FINISH",
                    "observation": "REFUSED: the report was empty. Write a real report after reading three pages, or choose READ.",
                })
                continue
            if len(pages_read) < MIN_PAGES_TO_READ:
                print(
                    f"STEP {step}  FINISH  ->  refused: only {len(pages_read)} of "
                    f"{MIN_PAGES_TO_READ} required pages read  (reason: {reason})"
                )
                state.append({
                    "action": "FINISH",
                    "summary": "FINISH",
                    "observation": f"REFUSED: only {len(pages_read)} of {MIN_PAGES_TO_READ} required pages read. Choose READ next.",
                })
                continue
            finished = True
            # All rules passed: accept the report, print the brief, and end.
            print(f"STEP {step}  FINISH  (reason: {reason})")
            print_brief(question, report, pages_read, also_found)
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

    return state, pages_read, also_found, report, finished, steps_used


def evaluate(question):
    # The five PASS/FAIL checks. They read the run's state and output - not
    # the model's opinion of itself. The score describes the RUN (did it
    # search, read several pages, finish in budget, recommend something, and
    # cite at least three sources), not whether the answer is factually
    # correct - none of these checks could know that.
    state, pages_read, also_found, report, finished, steps_used = run_agent(question)
    distinct_read = set(pages_read)

    c1 = any(entry["action"] == "SEARCH" for entry in state)
    c2 = len(distinct_read) > 1
    c3 = steps_used <= STEP_LIMIT
    c4 = bool(report) and "recommend" in report.lower()
    c5 = len(distinct_read) >= 3

    print("\n--- Evaluation ---")
    checks = [
        ("1. the search tool was used at least once", c1),
        ("2. more than one distinct source was consulted", c2),
        ("3. the run stayed within the step limit", c3),
        ("4. the brief contains a recommendation", c4),
        ("5. the brief lists at least three sources", c5),
    ]
    score = 0
    for label, passed in checks:
        print(("PASS" if passed else "FAIL") + "  " + label)
        if passed:
            score += 1
    print(f"Score: {score} of {len(checks)}")


def main():
    # Command-line entry point.
    #   python research_agent.py                  normal run (asks for a question)
    #   python research_agent.py --eval           normal run + five checks and score
    #   python research_agent.py --search "<query>"   test the search tool alone
    #   python research_agent.py --read <url>         test the page reader alone
    # The model settings are loaded from .env before anything that talks to
    # the model runs.
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

    if args and args[0] == "--eval":
        evaluate(question)
    else:
        run_agent(question)


if __name__ == "__main__":
    main()
