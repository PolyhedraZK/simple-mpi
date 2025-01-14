#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
from typing import List
from pydantic import BaseModel
from openai import OpenAI

class Violation(BaseModel):
    codeline: str
    reason: str

class StyleEvaluationResult(BaseModel):
    violations: List[Violation]
    score: int  # range from 0 to 100

def read_style_guide() -> str:
    """Read the style guide content."""
    with open('.github/workflows/style.md', 'r', encoding='utf-8') as f:
        return f.read()

def get_rust_files() -> List[str]:
    """Get all tracked Rust files in the repository."""
    result = subprocess.run(['git', 'ls-files', '*.rs'], 
                          capture_output=True, text=True, check=True)
    return sorted(result.stdout.splitlines())

def evaluate_file(client: OpenAI, style_guide: str, file_path: str) -> StyleEvaluationResult:
    """Evaluate a single Rust file against the style guide using OpenAI."""
    print(f"Evaluating {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # Read prompt template
    with open('.github/workflows/prompt.jinja', 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # Construct prompt that includes style guide, code, and uses the template
    prompt = f"""Style Guide:

                {style_guide}

                Code to evaluate:

                {code}

                {prompt_template}"""

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a Rust code style evaluator that checks code against style guidelines."},
            {"role": "user", "content": prompt}
        ],
        response_format=StyleEvaluationResult
    )

    # Parse JSON response into StyleEvaluationResult
    return response.choices[-1].message.parsed

def main():
    if 'OPENAI_API_KEY' not in os.environ:
        print("Error: OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    client = OpenAI()
    style_guide = read_style_guide()
    rust_files = get_rust_files()
    
    total_score = 0
    all_violations = []
    
    for file_path in rust_files:
        try:
            result = evaluate_file(client, style_guide, file_path)
            
            if result.violations:
                print(f"\nViolations in {file_path}:")
                for v in result.violations:
                    print(f"  Line: {v.codeline}")
                    print(f"  Reason: {v.reason}\n")
                all_violations.extend(result.violations)
            
            print(f"Score for {file_path}: {result.score}/100")
            total_score += result.score
            
        except Exception as e:
            print(f"Error evaluating {file_path}: {str(e)}")
            sys.exit(1)
    
    if not rust_files:
        print("No Rust files found to evaluate")
        sys.exit(0)

    average_score = total_score / len(rust_files)
    print(f"\nOverall average score: {average_score:.2f}/100")
    
    if average_score < 60:
        print("\nStyle check failed: Score below 60/100")
        sys.exit(1)
    
    if all_violations:
        print(f"\nTotal violations found: {len(all_violations)}")
        print("Note: Violations found but score is above threshold")
    else:
        print("\nNo style violations found!")
    
    sys.exit(0)

if __name__ == '__main__':
    main()
