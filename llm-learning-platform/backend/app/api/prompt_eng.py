"""
Prompt Engineering API Router
"""

from typing import Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.prompt_eng import list_templates, render_template, analyze_prompt, compare_prompts

router = APIRouter()


class RenderRequest(BaseModel):
    template_name: str = "zero_shot"
    variables: Dict[str, str] = {"question": "What is the capital of France?"}


class AnalyzeRequest(BaseModel):
    prompt: str = "Answer the following question: What is 2+2?"


class CompareRequest(BaseModel):
    prompts: List[str] = [
        "What is 2+2?",
        "Let's think step by step. What is 2+2?",
    ]


@router.get("/templates")
async def get_templates(category: Optional[str] = None):
    """List available prompt templates."""
    return {"templates": list_templates(category)}


@router.post("/render")
async def render(req: RenderRequest):
    """Render a prompt template with variables."""
    return render_template(req.template_name, req.variables)


@router.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Analyze a prompt's characteristics and get suggestions."""
    return analyze_prompt(req.prompt)


@router.post("/compare")
async def compare(req: CompareRequest):
    """Compare multiple prompt variants."""
    return compare_prompts(req.prompts)


@router.get("/techniques")
async def list_techniques():
    """List prompting techniques with descriptions."""
    return {
        "techniques": [
            {"name": "Zero-shot", "description": "Ask directly without examples", "difficulty": "beginner"},
            {"name": "Few-shot", "description": "Provide 1-5 examples before the query", "difficulty": "beginner"},
            {"name": "Chain-of-Thought", "description": "Add 'Let's think step by step' to elicit reasoning", "difficulty": "intermediate"},
            {"name": "Self-Consistency", "description": "Generate multiple reasoning paths and vote", "difficulty": "intermediate"},
            {"name": "Tree of Thoughts", "description": "Branching exploration with backtracking", "difficulty": "advanced"},
            {"name": "ReAct", "description": "Reason + Act loop with tool use", "difficulty": "advanced"},
            {"name": "Structured Output", "description": "Request JSON/XML formatted responses", "difficulty": "intermediate"},
            {"name": "Role Prompting", "description": "Assign a persona or expert role", "difficulty": "beginner"},
        ]
    }
