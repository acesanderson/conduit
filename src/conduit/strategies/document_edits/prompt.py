PROMPT_TEMPLATE = """\
<user_prompt>{{ user_prompt }}</user_prompt>

<document>{{ document }}</document>

Return ONLY valid JSON conforming to the DocumentEdits schema.
Apply edits sequentially; each search string must match the document \
as it exists after all prior edits have been applied.\
"""
