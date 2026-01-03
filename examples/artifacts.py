"""Artifacts Example - Logging Artifacts and Samples.

This example demonstrates:
- dn.log_input() / dn.log_output() - Log inputs and outputs
- dn.log_sample() - Log input/output pairs with metrics
- dn.log_artifact() - Log file artifacts
- dn.link_objects() - Link related objects together

Run with:
    python examples/artifacts.py
"""

import asyncio
import json
import tempfile
from pathlib import Path

import dreadnode as dn


async def main():
    # Configure SDK
    dn.configure(server="local")

    print("Artifacts Example")
    print("=" * 50)

    # Create a tracked run
    with dn.run("artifact-demo", params={"experiment": "v1", "model": "gpt-4"}):
        print("\n1. Logging inputs and outputs...")

        # Log individual inputs and outputs
        prompt = "What is the capital of France?"
        response = "The capital of France is Paris."

        dn.log_input("user_prompt", prompt)
        dn.log_output("model_response", response)

        # Link the response to the prompt (shows relationship in UI)
        dn.link_objects(response, prompt)
        print(f"   Logged prompt and response, linked together")

        # Log a sample (input/output pair with metrics)
        print("\n2. Logging samples with metrics...")

        samples = [
            {
                "prompt": "What is 2+2?",
                "response": "4",
                "correct": True,
                "latency_ms": 120,
            },
            {
                "prompt": "Capital of Germany?",
                "response": "Berlin",
                "correct": True,
                "latency_ms": 150,
            },
            {
                "prompt": "Largest ocean?",
                "response": "Pacific Ocean",
                "correct": True,
                "latency_ms": 180,
            },
        ]

        for i, sample in enumerate(samples):
            dn.log_sample(
                label=f"qa-sample-{i}",
                input=sample["prompt"],
                output=sample["response"],
                metrics={
                    "correct": 1.0 if sample["correct"] else 0.0,
                    "latency_ms": sample["latency_ms"],
                },
            )
            print(f"   Logged sample {i}: {sample['prompt'][:30]}...")

        # Log aggregate metrics
        avg_latency = sum(s["latency_ms"] for s in samples) / len(samples)
        accuracy = sum(1 for s in samples if s["correct"]) / len(samples)

        dn.log_metric("avg_latency_ms", avg_latency)
        dn.log_metric("accuracy", accuracy)
        print(f"\n   Aggregate metrics: accuracy={accuracy:.1%}, avg_latency={avg_latency:.0f}ms")

        # Log file artifacts
        print("\n3. Logging file artifacts...")

        # Create a temporary directory for artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save results as JSON
            results_path = tmpdir / "results.json"
            results = {
                "samples": samples,
                "metrics": {"accuracy": accuracy, "avg_latency_ms": avg_latency},
                "model": "gpt-4",
                "timestamp": "2024-01-15T10:30:00Z",
            }
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            dn.log_artifact(str(results_path), name="evaluation-results")
            print(f"   Logged artifact: results.json")

            # Save a predictions file
            predictions_path = tmpdir / "predictions.jsonl"
            with open(predictions_path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            dn.log_artifact(str(predictions_path), name="predictions")
            print(f"   Logged artifact: predictions.jsonl")

        print("\n" + "=" * 50)
        print("Artifacts example complete!")
        print("\nLogged:")
        print("  - 1 input/output pair (linked)")
        print(f"  - {len(samples)} samples with metrics")
        print("  - 2 file artifacts")


if __name__ == "__main__":
    asyncio.run(main())
