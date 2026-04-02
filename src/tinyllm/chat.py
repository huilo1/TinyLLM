from __future__ import annotations

CHAT_ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


def normalize_chat_role(role: str) -> str | None:
    role = str(role or "").strip().lower()
    if role in CHAT_ROLE_TOKENS:
        return role
    return None


def render_chat_message(role: str, content: str) -> str:
    normalized_role = normalize_chat_role(role)
    if normalized_role is None:
        raise ValueError(f"Unsupported chat role: {role!r}")

    content = str(content or "").strip()
    if not content:
        return CHAT_ROLE_TOKENS[normalized_role]
    return f"{CHAT_ROLE_TOKENS[normalized_role]}\n{content}"


def build_chat_transcript(
    messages: list[dict],
    separator: str = "\n\n",
    add_generation_prompt: bool = False,
) -> str:
    blocks: list[str] = []
    for message in messages:
        role = normalize_chat_role(message.get("role", ""))
        if role is None:
            continue
        content = str(message.get("content", "") or "").strip()
        if not content:
            continue
        blocks.append(render_chat_message(role, content))

    if add_generation_prompt:
        blocks.append(CHAT_ROLE_TOKENS["assistant"])
    return separator.join(blocks)


def extract_assistant_reply(text: str) -> str:
    reply = str(text or "").lstrip()
    cut_positions = [
        reply.find(role_token)
        for role_token in CHAT_ROLE_TOKENS.values()
        if reply.find(role_token) > 0
    ]
    if cut_positions:
        reply = reply[: min(cut_positions)]
    return reply.strip()
