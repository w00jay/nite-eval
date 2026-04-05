"""Centralized rubric definitions for judge-scored dimensions.

Used by the orchestrator, calibration scripts, and validation scripts.
Each rubric maps a scoring dimension name to a ternary (1/3/5) description.
"""

# Standard rubrics used across the calibration and eval pipeline.
# Keys match the dimension names in task scoring configs.
JUDGE_RUBRICS: dict[str, str] = {
    "coverage": (
        "1 (Poor): Misses most aspects of the question\n"
        "3 (Acceptable): Covers most aspects adequately\n"
        "5 (Excellent): Comprehensive, addresses all aspects with depth"
    ),
    "synthesis": (
        "1 (Poor): Pure list of facts, no connections drawn\n"
        "3 (Acceptable): Some meaningful connections between ideas\n"
        "5 (Excellent): Excellent synthesis with original cross-cutting insights"
    ),
    "accuracy": (
        "1 (Poor): Major factual errors or hallucinations\n"
        "3 (Acceptable): Mostly accurate with minor issues\n"
        "5 (Excellent): Fully accurate, no hallucinations, properly caveated"
    ),
    "recommendation_quality": (
        "1 (Poor): No clear recommendation or generic platitude\n"
        "3 (Acceptable): Clear recommendation with basic reasoning\n"
        "5 (Excellent): Specific, well-reasoned, addresses constraints, actionable"
    ),
    "specificity": (
        "1 (Poor): Entirely vague ('set up backend')\n"
        "3 (Acceptable): Mix of specific and general steps\n"
        "5 (Excellent): All steps are concrete with clear deliverables"
    ),
    "risk_awareness": (
        "1 (Poor): No risks mentioned\n"
        "3 (Acceptable): Identifies key risks with basic mitigation\n"
        "5 (Excellent): Thorough risks, mitigations, fallbacks, rollback plan"
    ),
    "completeness": (
        "1 (Poor): Major gaps in coverage\n"
        "3 (Acceptable): Covers main phases adequately\n"
        "5 (Excellent): Complete with all phases, dependencies, edge cases"
    ),
    "dependency_correctness": (
        "1 (Poor): Dependencies wrong or ignored\n"
        "3 (Acceptable): Mostly correct ordering\n"
        "5 (Excellent): Perfect ordering with explicit dependency justification"
    ),
    "feasibility": (
        "1 (Poor): Unrealistic or ignores constraints\n"
        "3 (Acceptable): Feasible with some optimistic estimates\n"
        "5 (Excellent): Pragmatic, right-sized, acknowledges trade-offs"
    ),
    "code_quality": (
        "1 (Poor): No types, bad naming, no structure\n"
        "3 (Acceptable): Types present, reasonable structure\n"
        "5 (Excellent): Production-quality, documented, testable, idiomatic"
    ),
    "error_handling": (
        "1 (Poor): No error handling\n"
        "3 (Acceptable): Handles main error cases\n"
        "5 (Excellent): Specific error types, logging, graceful degradation"
    ),
    "architecture": (
        "1 (Poor): Monolithic, no separation\n"
        "3 (Acceptable): Reasonable structure\n"
        "5 (Excellent): Composable, extensible, right level of abstraction"
    ),
    "reasoning_quality": (
        "1 (Poor): No reasoning, just restates data\n"
        "3 (Acceptable): Adequate reasoning with some logic\n"
        "5 (Excellent): Explicit reasoning chain, considers alternatives, well-justified"
    ),
    "practical_output": (
        "1 (Poor): Generic or unhelpful output\n"
        "3 (Acceptable): Useful with some specifics\n"
        "5 (Excellent): Directly actionable with concrete next steps"
    ),
    "technical_depth": (
        "1 (Poor): Surface-level, no understanding of trade-offs\n"
        "3 (Acceptable): Demonstrates basic understanding with some reasoning\n"
        "5 (Excellent): Deep architectural understanding with nuanced trade-offs"
    ),
    "practical_judgment": (
        "1 (Poor): Recommendations ignore real-world constraints\n"
        "3 (Acceptable): Addresses main constraints with reasonable suggestions\n"
        "5 (Excellent): Pragmatic, specific to user's context, acknowledges trade-offs"
    ),
    "structure": (
        "1 (Poor): Disorganized wall of text\n"
        "3 (Acceptable): Basic organization with sections\n"
        "5 (Excellent): Well-structured with clear flow and easy to follow"
    ),
    "actionability": (
        "1 (Poor): Generic advice, no specific next steps\n"
        "3 (Acceptable): Some actionable suggestions\n"
        "5 (Excellent): Specific, implementable recommendations with clear priorities"
    ),
    "phased_approach": (
        "1 (Poor): No phases, monolithic plan\n"
        "3 (Acceptable): Basic phasing with some logic\n"
        "5 (Excellent): Clear phases with dependencies, parallel-run, and rollback"
    ),
    "risk_mitigation": (
        "1 (Poor): No risks identified\n"
        "3 (Acceptable): Major risks identified with basic mitigation\n"
        "5 (Excellent): Comprehensive risks with specific mitigations and fallbacks"
    ),
    "architecture_quality": (
        "1 (Poor): No clear architecture, muddled responsibilities\n"
        "3 (Acceptable): Reasonable separation of concerns\n"
        "5 (Excellent): Clean layers, clear data flow, appropriate technology choices"
    ),
    "scalability_awareness": (
        "1 (Poor): No consideration of growth\n"
        "3 (Acceptable): Mentions scalability concerns\n"
        "5 (Excellent): Pragmatic growth path with clear upgrade triggers"
    ),
    "edge_case_handling": (
        "1 (Poor): No edge cases considered\n"
        "3 (Acceptable): Handles common edge cases\n"
        "5 (Excellent): Comprehensive edge case handling with graceful degradation"
    ),
    "cache_design": (
        "1 (Poor): No caching or broken cache logic\n"
        "3 (Acceptable): Basic TTL caching works\n"
        "5 (Excellent): Correct TTL, thread-safe, no memory leaks"
    ),
    "answer_quality": (
        "1 (Poor): Wrong or unusable answer\n"
        "3 (Acceptable): Correct with basic explanation\n"
        "5 (Excellent): Clear, specific, with practical advice"
    ),
    "context_assembly": (
        "1 (Poor): Hallucinated data, didn't use tool results\n"
        "3 (Acceptable): Used most tool results correctly\n"
        "5 (Excellent): Systematic data gathering then reasoning, no hallucinations"
    ),
    "analysis_structure": (
        "1 (Poor): Unstructured, missing key sections\n"
        "3 (Acceptable): Basic structure with main sections\n"
        "5 (Excellent): Systematic process, each section references specific data"
    ),
    "judgment_quality": (
        "1 (Poor): No clear recommendation or ignores conflicting signals\n"
        "3 (Acceptable): Clear recommendation with basic reasoning\n"
        "5 (Excellent): Addresses conflicting signals, states confidence, specific levels"
    ),
    "diagnostic_approach": (
        "1 (Poor): Blind retry or random actions\n"
        "3 (Acceptable): Some investigation before fixing\n"
        "5 (Excellent): Systematic diagnosis: check health, read logs, identify root cause"
    ),
    "communication": (
        "1 (Poor): No explanation of what happened\n"
        "3 (Acceptable): Basic explanation of the fix\n"
        "5 (Excellent): Clear explanation of root cause, fix applied, and prevention"
    ),
}


def get_rubric(dimension: str) -> str:
    """Get rubric for a dimension, with a generic fallback."""
    return JUDGE_RUBRICS.get(dimension, f"Rate {dimension} from 1 (poor) to 5 (excellent)")
