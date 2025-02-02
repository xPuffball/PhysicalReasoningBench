# PhysicalReasoningBench

A benchmark for evaluating physical reasoning capabilities in Large Language Models through escape room puzzles.

## Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/physical-reasoning-bench.git
cd physical-reasoning-bench
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up API keys
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Running the Benchmark

1. Run the benchmark:
```python
python src/run_benchmark.py
```

2. Evaluate results:
```python
python src/evaluate_results.py
```

3. Generate visualizations:
```python
python src/visualize_results.py
```

## Benchmark Problems

The benchmark consists of five physical reasoning problems:
1. Fluid Displacement (retrieving a ping pong ball)
2. UV Detection (finding fingerprints)
3. Cipher Mechanism (decoding a message)
4. Vacuum Extraction (removing a stuck can)
5. Collaborative Retrieval (prison cell puzzle)

Each problem tests different aspects of physical reasoning including:
- Understanding of physical principles
- Step-by-step solution planning
- Handling of irrelevant information

## Output

The benchmark produces:
- Raw model responses (`results/benchmark_results.json`)
- Evaluation scores (`results/benchmark_evaluations.json`)
- Visualization plots (`results/figures/`)