"""
Prompt Engineering Engine

Implements prompt templates, CoT, few-shot patterns, and prompt evaluation.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class PromptTemplate:
    name: str
    description: str
    template: str
    category: str
    variables: List[str] = field(default_factory=list)


# Built-in prompt templates
PROMPT_TEMPLATES = [
    PromptTemplate(
        name="zero_shot",
        description="Direct question without examples",
        template="Answer the following question:\n\n{question}\n\nAnswer:",
        category="basic",
        variables=["question"],
    ),
    PromptTemplate(
        name="few_shot",
        description="Few-shot learning with examples",
        template="Here are some examples:\n\n{examples}\n\nNow answer:\n{question}\n\nAnswer:",
        category="basic",
        variables=["examples", "question"],
    ),
    PromptTemplate(
        name="chain_of_thought",
        description="Step-by-step reasoning",
        template="Question: {question}\n\nLet's think step by step:\n",
        category="reasoning",
        variables=["question"],
    ),
    PromptTemplate(
        name="self_consistency",
        description="Multiple reasoning paths for voting",
        template="Question: {question}\n\nLet's solve this step by step. I'll explore multiple approaches:\n\nApproach 1:\n",
        category="reasoning",
        variables=["question"],
    ),
    PromptTemplate(
        name="tree_of_thoughts",
        description="Branching reasoning with evaluation",
        template="Problem: {question}\n\nExplore different solution paths:\n\nPath A:\nThought 1.1: ",
        category="advanced",
        variables=["question"],
    ),
    PromptTemplate(
        name="system_user_assistant",
        description="Chat-style with system prompt",
        template="<|system|>\n{system_prompt}\n<|user|>\n{user_message}\n<|assistant|>\n",
        category="chat",
        variables=["system_prompt", "user_message"],
    ),
    PromptTemplate(
        name="structured_output",
        description="Request JSON/structured output",
        template='Answer the question in valid JSON format.\n\nQuestion: {question}\n\nRespond with a JSON object containing "answer" and "confidence" fields:\n',
        category="structured",
        variables=["question"],
    ),
    PromptTemplate(
        name="role_play",
        description="Assign a persona/role",
        template="You are {role}. {context}\n\nUser: {question}\n\nResponse:",
        category="chat",
        variables=["role", "context", "question"],
    ),
]


def list_templates(category: Optional[str] = None) -> List[Dict]:
    """List available prompt templates."""
    templates = PROMPT_TEMPLATES
    if category:
        templates = [t for t in templates if t.category == category]
    return [
        {
            "name": t.name,
            "description": t.description,
            "template": t.template,
            "category": t.category,
            "variables": t.variables,
        }
        for t in templates
    ]


def render_template(template_name: str, variables: Dict[str, str]) -> Dict:
    """Render a prompt template with variables."""
    template = next((t for t in PROMPT_TEMPLATES if t.name == template_name), None)
    if not template:
        return {"error": f"Template '{template_name}' not found"}

    rendered = template.template
    missing = []
    for var in template.variables:
        if var in variables:
            rendered = rendered.replace(f"{{{var}}}", variables[var])
        else:
            missing.append(var)

    return {
        "template_name": template_name,
        "rendered": rendered,
        "missing_variables": missing,
        "token_count_estimate": len(rendered.split()),
        "character_count": len(rendered),
    }


def analyze_prompt(prompt: str) -> Dict:
    """Analyze a prompt's characteristics."""
    lines = prompt.strip().split("\n")
    words = prompt.split()

    # Count few-shot examples (heuristic: look for numbered items or Q/A pairs)
    example_count = sum(1 for line in lines if line.strip().startswith(("Example", "Q:", "Input:", "1.", "2.", "3.")))

    # Check for CoT markers
    has_cot = any(phrase in prompt.lower() for phrase in [
        "step by step", "let's think", "reasoning:", "first,", "therefore,"
    ])

    # Check for structured output request
    has_structured = any(phrase in prompt.lower() for phrase in ["json", "format:", "schema:", "output format"])

    # Check for system/role
    has_role = any(phrase in prompt.lower() for phrase in ["you are", "act as", "role:", "persona:"])

    return {
        "character_count": len(prompt),
        "word_count": len(words),
        "line_count": len(lines),
        "estimated_tokens": int(len(words) * 1.3),
        "num_examples": example_count,
        "techniques_detected": {
            "few_shot": example_count > 0,
            "chain_of_thought": has_cot,
            "structured_output": has_structured,
            "role_assignment": has_role,
            "multi_turn": "<|user|>" in prompt or "User:" in prompt,
        },
        "suggestions": _generate_suggestions(prompt, has_cot, example_count, has_structured),
    }


def _generate_suggestions(prompt: str, has_cot: bool, examples: int, has_structured: bool) -> List[str]:
    suggestions = []
    if len(prompt) < 50:
        suggestions.append("Consider adding more context or instructions to improve output quality.")
    if not has_cot and len(prompt.split()) > 20:
        suggestions.append("Try adding 'Let's think step by step' for complex reasoning tasks.")
    if examples == 0:
        suggestions.append("Adding 1-3 examples (few-shot) can significantly improve accuracy.")
    if not has_structured:
        suggestions.append("For structured data, specify the output format (JSON, table, list).")
    if "please" not in prompt.lower() and "must" not in prompt.lower():
        suggestions.append("Being specific about requirements (e.g., 'You must...') improves compliance.")
    return suggestions


def compare_prompts(prompts: List[str]) -> Dict:
    """Compare multiple prompt variants."""
    analyses = []
    for i, prompt in enumerate(prompts):
        analysis = analyze_prompt(prompt)
        analysis["variant"] = i + 1
        analysis["prompt_preview"] = prompt[:100] + ("..." if len(prompt) > 100 else "")
        analyses.append(analysis)

    return {"num_variants": len(prompts), "analyses": analyses}
