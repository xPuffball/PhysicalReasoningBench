import os
import json
import asyncio
import aiohttp
from typing import Dict, List
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential

@dataclass
class JudgingCriteria:
    problem_id: str
    scenario: str
    available_items: List[str]
    constraints: List[str]
    correct_principles: List[str]
    solution_steps: List[str]
    red_herrings: List[str]
    model_response: str

class BenchmarkJudge:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        
    def _create_evaluation_prompt(self, criteria: JudgingCriteria) -> str:
        return f"""You are an expert judge evaluating an LLM's understanding of physical reasoning puzzles. You will evaluate the response based on three criteria:

1. Physical Understanding Score (0-2):
- How well does the response understand and explain the relevant physical principles?
- Are the principles correctly applied to the solution?

2. Solution Path Score (0-2):
- Are all necessary steps present and in correct order?
- Is the solution practical and would it work given the constraints?

3. Red Herring Handling (0-1):
- Does the response avoid using irrelevant items?
- Does it explicitly recognize which items are not needed?

Original Problem:
Scenario: {criteria.scenario}

Available Items:
{chr(10).join(f"- {item}" for item in criteria.available_items)}

Constraints:
{chr(10).join(f"- {constraint}" for constraint in criteria.constraints)}

Correct Physical Principles:
{chr(10).join(f"- {principle}" for principle in criteria.correct_principles)}

Expected Solution Steps:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(criteria.solution_steps))}

Red Herring Items:
{chr(10).join(f"- {item}" for item in criteria.red_herrings)}

Model's Response:
{criteria.model_response}

Provide your evaluation in the following JSON format:
{{
    "physical_understanding_score": float,  # Score from 0-2
    "physical_understanding_explanation": "string",  # Detailed explanation
    "solution_path_score": float,  # Score from 0-2
    "solution_path_explanation": "string",  # Detailed explanation
    "red_herring_score": int,  # Score 0 or 1
    "red_herring_explanation": "string",  # Detailed explanation
    "overall_feedback": "string"  # General feedback and suggestions
}}"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def evaluate_response(self, criteria: JudgingCriteria) -> Dict:
        """Evaluate a model's response using GPT-4 as judge."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert evaluator of physical reasoning puzzles."},
                        {"role": "user", "content": self._create_evaluation_prompt(criteria)}
                    ],
                    "temperature": 0.2
                }
            ) as response:
                if response.status != 200:
                    raise ValueError(f"API request failed: {await response.text()}")
                
                result = await response.json()
                try:
                    evaluation = json.loads(result["choices"][0]["message"]["content"])
                    return evaluation
                except (KeyError, json.JSONDecodeError) as e:
                    raise ValueError(f"Failed to parse evaluation: {e}")

class ResultsEvaluator:
    """Evaluates benchmark results using LLM judge."""
    
    def __init__(self, problems_dict: Dict, results_file: str, judge: BenchmarkJudge):
        self.problems = problems_dict
        self.results_file = results_file
        self.judge = judge
    
    def load_results(self) -> Dict:
        """Load results from file."""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def _create_judging_criteria(self, problem_id: str, response: str) -> JudgingCriteria:
        """Create judging criteria from problem and response."""
        problem = self.problems[problem_id]
        return JudgingCriteria(
            problem_id=problem_id,
            scenario=problem["setup"]["scenario"],
            available_items=problem["setup"]["available_items"],
            constraints=problem["setup"]["constraints"],
            correct_principles=problem["physical_principles"],
            solution_steps=problem["solution"]["steps"],
            red_herrings=problem["red_herrings"],
            model_response=response
        )
    
    async def evaluate_all(self) -> Dict:
        """Evaluate all results using the judge."""
        results = self.load_results()
        evaluations = {
            "gpt4_evaluations": [],
            "claude_evaluations": []
        }
        
        # Evaluate GPT-4 results
        for result in results["gpt4_results"]:
            if result["response"]:  # Only evaluate if there's a response
                criteria = self._create_judging_criteria(
                    result["problem_id"],
                    result["response"]
                )
                evaluation = await self.judge.evaluate_response(criteria)
                evaluations["gpt4_evaluations"].append({
                    "problem_id": result["problem_id"],
                    "evaluation": evaluation
                })
        
        # Evaluate Claude results
        for result in results["claude_results"]:
            if result["response"]:  # Only evaluate if there's a response
                criteria = self._create_judging_criteria(
                    result["problem_id"],
                    result["response"]
                )
                evaluation = await self.judge.evaluate_response(criteria)
                evaluations["claude_evaluations"].append({
                    "problem_id": result["problem_id"],
                    "evaluation": evaluation
                })
        
        return evaluations

    def calculate_scores(self, evaluations: Dict) -> Dict:
        """Calculate aggregate scores from evaluations."""
        scores = {
            "gpt4": {
                "overall_score": 0,
                "physical_understanding_avg": 0,
                "solution_path_avg": 0,
                "red_herring_avg": 0,
                "problem_scores": {}
            },
            "claude": {
                "overall_score": 0,
                "physical_understanding_avg": 0,
                "solution_path_avg": 0,
                "red_herring_avg": 0,
                "problem_scores": {}
            }
        }
        
        # Process GPT-4 evaluations
        if evaluations["gpt4_evaluations"]:
            gpt4_scores = scores["gpt4"]
            for eval_result in evaluations["gpt4_evaluations"]:
                evaluation = eval_result["evaluation"]
                problem_id = eval_result["problem_id"]
                
                # Calculate problem score
                problem_score = (
                    evaluation["physical_understanding_score"] +
                    evaluation["solution_path_score"] +
                    evaluation["red_herring_score"]
                )
                gpt4_scores["problem_scores"][problem_id] = problem_score
                
                # Update averages
                gpt4_scores["physical_understanding_avg"] += evaluation["physical_understanding_score"]
                gpt4_scores["solution_path_avg"] += evaluation["solution_path_score"]
                gpt4_scores["red_herring_avg"] += evaluation["red_herring_score"]
            
            # Calculate final averages
            num_problems = len(evaluations["gpt4_evaluations"])
            gpt4_scores["physical_understanding_avg"] /= num_problems
            gpt4_scores["solution_path_avg"] /= num_problems
            gpt4_scores["red_herring_avg"] /= num_problems
            gpt4_scores["overall_score"] = (
                gpt4_scores["physical_understanding_avg"] +
                gpt4_scores["solution_path_avg"] +
                gpt4_scores["red_herring_avg"]
            )
        
        # Process Claude evaluations (same logic as GPT-4)
        if evaluations["claude_evaluations"]:
            claude_scores = scores["claude"]
            for eval_result in evaluations["claude_evaluations"]:
                evaluation = eval_result["evaluation"]
                problem_id = eval_result["problem_id"]
                
                problem_score = (
                    evaluation["physical_understanding_score"] +
                    evaluation["solution_path_score"] +
                    evaluation["red_herring_score"]
                )
                claude_scores["problem_scores"][problem_id] = problem_score
                
                claude_scores["physical_understanding_avg"] += evaluation["physical_understanding_score"]
                claude_scores["solution_path_avg"] += evaluation["solution_path_score"]
                claude_scores["red_herring_avg"] += evaluation["red_herring_score"]
            
            num_problems = len(evaluations["claude_evaluations"])
            claude_scores["physical_understanding_avg"] /= num_problems
            claude_scores["solution_path_avg"] /= num_problems
            claude_scores["red_herring_avg"] /= num_problems
            claude_scores["overall_score"] = (
                claude_scores["physical_understanding_avg"] +
                claude_scores["solution_path_avg"] +
                claude_scores["red_herring_avg"]
            )
        
        return scores

async def run_evaluations():
    # Load API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OpenAI API key must be set in environment variables")
    
    # Initialize judge
    judge = BenchmarkJudge(api_key=openai_key)
    
    # Initialize evaluator
    evaluator = ResultsEvaluator(
        problems_dict=problems_dict,
        results_file="benchmark_results.json",
        judge=judge
    )
    
    try:
        # Run evaluations
        print("Evaluating benchmark results...")
        evaluations = await evaluator.evaluate_all()
        
        # Calculate scores
        print("Calculating scores...")
        scores = evaluator.calculate_scores(evaluations)
        
        # Save evaluations and scores
        with open("benchmark_evaluations.json", "w") as f:
            json.dump({
                "evaluations": evaluations,
                "scores": scores
            }, f, indent=2)
        
        print("\nEvaluation completed. Results saved to benchmark_evaluations.json")
        
        # Print summary
        print("\nSummary:")
        print("GPT-4 Overall Score:", scores["gpt4"]["overall_score"])
        print("Claude Overall Score:", scores["claude"]["overall_score"])
    
    except Exception as e:
        print(f"Error running evaluations: {str(e)}")
        raise

await run_evaluations()