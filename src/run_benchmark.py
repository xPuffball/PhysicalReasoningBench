import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential

with open('problems.json', 'r') as f:
    problems_dict = json.load(f)

class ModelInterface:
    """Interface for different LLM APIs."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

class GPT4Interface(ModelInterface):
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_response(self, prompt: str) -> str:
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")
        
        try:
            async with self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"GPT-4 API request failed: {error_text}")
                
                result = await response.json()
                if "choices" not in result or not result["choices"]:
                    raise ValueError(f"Unexpected GPT-4 API response format: {result}")
                
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error in GPT-4 API call: {str(e)}")
            raise

class ClaudeInterface(ModelInterface):
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_response(self, prompt: str) -> str:
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")
        
        try:
            async with self.session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "anthropic-version": "2023-06-01",
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Claude API request failed: {error_text}")
                
                result = await response.json()
                if "content" not in result or not result.get("content"):
                    raise ValueError(f"Unexpected Claude API response format: {result}")
                
                return result["content"][0]["text"]
        except Exception as e:
            print(f"Error in Claude API call: {str(e)}")
            raise

class BenchmarkRunner:
    """Runs the benchmark across multiple models and problems."""
    
    def __init__(
        self,
        problems: Dict,
        openai_key: str,
        anthropic_key: str
    ):
        """Initialize the benchmark runner.
        
        Args:
            problems: Dictionary of benchmark problems
            openai_key: OpenAI API key
            anthropic_key: Anthropic API key
        """
        self.problems = problems
        self.openai_key = openai_key
        self.anthropic_key = anthropic_key
    
    def _create_prompt(self, problem: Dict) -> str:
        """Create the prompt for a model from a problem."""
        return f"""You are solving an escape room puzzle. Here is the scenario:

{problem["setup"]["scenario"]}

Available items:
{chr(10).join('- ' + item for item in problem["setup"]["available_items"])}

Constraints:
{chr(10).join('- ' + constraint for constraint in problem["setup"]["constraints"])}

Provide a solution that:
1. Explains the physical principles involved
2. Lists step-by-step actions
3. Identifies which items are needed and why

Your response should be structured as:
Physical Principles:
[Your explanation]

Solution Steps:
[Your steps]

Required Items:
[Your list]"""

    async def evaluate_model(
        self,
        model_interface: ModelInterface,
        problem_id: str
    ) -> Dict:
        """Evaluate a single model on a single problem."""
        problem = self.problems[problem_id]
        prompt = self._create_prompt(problem)
        
        try:
            model_response = await model_interface.generate_response(prompt)
            return {
                "problem_id": problem_id,
                "response": model_response
            }
        except Exception as e:
            print(f"Error evaluating problem {problem_id}: {str(e)}")
            return {
                "problem_id": problem_id,
                "response": None,
                "error": str(e)
            }

    async def run_benchmark(
        self,
        problem_ids: Optional[List[str]] = None
    ) -> Dict:
        """Run the complete benchmark."""
        problem_ids = problem_ids or list(self.problems.keys())
        results = {
            "gpt4_results": [],
            "claude_results": []
        }
        
        # Run GPT-4 evaluations
        async with GPT4Interface(self.openai_key) as gpt4:
            for problem_id in problem_ids:
                result = await self.evaluate_model(gpt4, problem_id)
                results["gpt4_results"].append(result)
        
        # Run Claude evaluations
        async with ClaudeInterface(self.anthropic_key) as claude:
            for problem_id in problem_ids:
                result = await self.evaluate_model(claude, problem_id)
                results["claude_results"].append(result)
        
        return results

async def main():
    # Load API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key or not anthropic_key:
        raise ValueError("API keys must be set in environment variables")
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(
        problems=problems_dict,
        openai_key=openai_key,
        anthropic_key=anthropic_key
    )
    
    try:
        # Run benchmark for all problems
        results = await runner.run_benchmark()
        
        # Save results
        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("Benchmark completed. Results saved to benchmark_results.json")
        
        # Print first response as example
        if results["gpt4_results"] and results["gpt4_results"][0]["response"]:
            print("\nExample GPT-4 response for first problem:")
            print(results["gpt4_results"][0]["response"])
    
    except Exception as e:
        print(f"Error running benchmark: {str(e)}")
        raise

await main()