"""Prompt text and chat-format rendering.

Mack renders turns itself instead of re-running the chat template over the whole
history, because the live cache can only be appended to (the recurrent layers of
a hybrid model cannot be trimmed). The invariant is that the cached context always
ends right after an end-of-turn token.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM_PROMPT = """You are Mack, a local AI assistant that runs entirely on this Mac. You were built as the Mack.ai project to be useful without slowing down the rest of the computer. Be direct, accurate and concise.

In a long conversation, older messages are compacted to save memory. When that has happened, a section titled "Memory of the earlier conversation" appears below with exact quotes and a short summary, and a user message may start with original text recalled from earlier.

Rules:
- Answer the latest message directly. Do not open with a recap of what you remember, and do not write your own "Recalled" or "Memory" headings.
- For anything about earlier in the conversation, rely only on what you can actually see: the messages, the Memory section, and recalled text. If the detail is not there, say you can't see that part of the conversation anymore and ask the user to repeat it. Never guess a number, name, date or decision."""

MEMORY_HEADER = "## Memory of the earlier conversation"
MEMORY_NOTE = (
    "Messages before this point were compacted to save memory. The quotes below are exact. "
    "Anything not listed here or visible below is no longer visible to you."
)

SUMMARY_INSTRUCTION = """Summarize the conversation excerpt below as short factual bullet points.

Rules:
- One fact per bullet, each starting with "- ".
- Use only facts stated in the excerpt. Copy numbers, names and dates exactly.
- Say who said or wants what ("The user ...", "Mack ...").
- Skip greetings and small talk. At most {max_bullets} bullets. No introduction and no closing line.

Excerpt:
{excerpt}"""

RECALL_OPEN = "[Recalled: original text from earlier in this conversation]"
RECALL_CLOSE = "[End of recalled text]"


def memory_block(pins: list, summary: list[str]) -> str:
    if not pins and not summary:
        return ""
    lines = [MEMORY_HEADER, MEMORY_NOTE]
    if pins:
        lines.append("\nExact quotes:")
        for p in pins:
            who = "User" if p.role == "user" else "Mack"
            lines.append(f'- {who} (message {p.turn + 1}): "{p.quote}"')
    if summary:
        lines.append("\nSummary of the rest:")
        lines.extend(f"- {s}" for s in summary)
    return "\n".join(lines)


def recall_block(hits: list) -> str:
    lines = [RECALL_OPEN]
    for h in hits:
        who = "User" if h.role == "user" else "Mack"
        lines.append(f"{who} (message {h.turn + 1}): {h.text}")
    lines.append(RECALL_CLOSE)
    return "\n".join(lines)


@dataclass
class ChatFormat:
    """Hand-rendered turn markup for the two families Mack supports."""

    system_open: str
    user_open: str
    assistant_open: str
    close: str
    sep: str  # between a closed turn and the next opened one

    def system(self, text: str) -> str:
        return f"{self.system_open}{text}{self.close}"

    def user(self, text: str) -> str:
        return f"{self.sep}{self.user_open}{text}{self.close}"

    def assistant(self, text: str) -> str:
        return f"{self.sep}{self.assistant_open}{text}{self.close}"

    def generation_prompt(self) -> str:
        return f"{self.sep}{self.assistant_open}"


def detect_format(tokenizer, enable_thinking: bool = False) -> ChatFormat:
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )
    if "<|im_start|>" in probe:
        # Anything the template puts after the assistant header (Qwen's empty
        # <think> block when thinking is off) is part of how every reply starts.
        tail = probe.rsplit("<|im_start|>assistant\n", 1)[1]
        return ChatFormat(
            system_open="<|im_start|>system\n",
            user_open="<|im_start|>user\n",
            assistant_open="<|im_start|>assistant\n" + tail,
            close="<|im_end|>",
            sep="\n",
        )
    if "<|start_header_id|>" in probe:
        tail = probe.rsplit("<|start_header_id|>assistant<|end_header_id|>\n\n", 1)[1]
        return ChatFormat(
            system_open="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
            user_open="<|start_header_id|>user<|end_header_id|>\n\n",
            assistant_open="<|start_header_id|>assistant<|end_header_id|>\n\n" + tail,
            close="<|eot_id|>",
            sep="",
        )
    raise ValueError("unsupported chat template: Mack renders ChatML (Qwen) and Llama-3 formats")
