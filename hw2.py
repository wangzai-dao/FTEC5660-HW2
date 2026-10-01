#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any
from langchain_deepseek import ChatDeepSeek
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()


def build_agent(tools: list[Any]) -> Any:
    """Create and return your agent once.

    ``tools`` are the six SocialGraph MCP tools (Facebook + LinkedIn search and
    profile lookup), already wrapped as LangChain tools. You may add your own
    local tools as well.
    """
    model = ChatDeepSeek(
        model=MODEL_NAME,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0,
        timeout=120,
        max_retries=2,
    )

    system_prompt = """You are a meticulous CV verification agent. Your job is to verify the claims in a candidate's CV against their LinkedIn and Facebook profiles using the provided SocialGraph MCP tools.

IMPORTANT RULES:
- LinkedIn is the primary source of truth. Facebook should agree with LinkedIn.
- Discrepancies can ONLY be in these fields: name, city, jobs (company, title, seniority, start and end years), education (degree, school, field, graduation year), and skills.
- Wording differences are NOT discrepancies. Examples: "Bachelor of Science" vs "BSc", "UI/UX Design" vs "UI/UX", "Senior Engineer" for an Engineer role with seniority "senior", or listing fewer skills than the profile.
- A discrepancy means at least one false claim: inflated job title, shifted employment or graduation years, upgraded degree, fake school or employer, wrong location, or a skill the candidate does not have.
- Job descriptions, headline, and hometown are NEVER sources of discrepancy.
- Many candidates share the same name. You MUST verify the right person by using additional information from the CV (location, company, education, skills) to narrow down search results. Do not assume the first search result is the correct person.
- Use the tools to search and retrieve profiles. Start by searching LinkedIn with the candidate's name and location. If many results, narrow down by company or education. Retrieve the full LinkedIn profile. You may also check Facebook if needed.
- After careful verification, output a single JSON object with exactly two keys: "score" (float between 0 and 1) and "reason" (string).
  - Score > 0.5 means the CV is valid (no discrepancy).
  - Score <= 0.5 means the CV has at least one discrepancy.
  - If all claims match, give a high score (e.g., 0.9-1.0).
  - If any discrepancy is found, give a low score (e.g., 0.0-0.2).
  - If you cannot verify (e.g., cannot find the person), give 0.5.
- Your final answer MUST be only the JSON object, with no other text.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "CV text:\n{input}\n\nVerify this CV and output the JSON."),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(model, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=20,
        return_intermediate_steps=False,
    )
async def _score_one(agent: Any, filename: str, text: str, sem: asyncio.Semaphore) -> tuple[str, float]:
    """异步处理单个 CV，带并发信号量。"""
    async with sem:
        try:
            result = await agent.ainvoke({"input": text})
            output = result.get("output", "")

            # 尝试提取 JSON
            match = re.search(r'\{[^{}]*"score"[^{}]*\}', output, re.DOTALL)
            if not match:
                match = re.search(r'\{.*?"score".*?\}', output, re.DOTALL)

            if match:
                try:
                    data = json.loads(match.group())
                    score = float(data["score"])
                    if 0.0 <= score <= 1.0:
                        return filename, score
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass

            # 回退：从输出中找第一个 0~1 的浮点数
            nums = re.findall(r'\b(0\.\d+|1\.0|0|1)\b', output)
            if nums:
                score = float(nums[0])
                if 0.0 <= score <= 1.0:
                    return filename, score

            return filename, 0.5  # 无法解析时给中性分

        except Exception as e:
            print(f"[ERROR] Failed to score {filename}: {e}")
            return filename, 0.5


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Run your agent and return one reliability score per CV.

    ``cvs`` maps each file name to its text, e.g. ``{"CV_1.pdf": "...", ...}``.
    Return a float in [0, 1] for every file name: higher means the CV is more
    likely consistent with the candidate's LinkedIn/Facebook data. A score
    above 0.5 counts as "valid", 0.5 or below counts as "has discrepancy".

        {"CV_1.pdf": 0.9, "CV_4.pdf": 0.1, ...}

    Catch errors per CV (e.g. a failed API call) and still return a score for
    it: an exception here means no results.csv, which scores zero.

    MCP tools are async, so call your agent with ``await agent.ainvoke(...)``.
    You may verify CVs in parallel (e.g. ``asyncio.gather``), but keep at most
    about 3 CVs in flight (e.g. with ``asyncio.Semaphore(3)``): the MCP server is
    shared by the whole class.
    """
    sem = asyncio.Semaphore(3)
    tasks = [_score_one(agent, name, text, sem) for name, text in cvs.items()]
    results = await asyncio.gather(*tasks)
    return dict(results)
def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())
