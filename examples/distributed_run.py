"""Distributed Run Example - Context Transfer.

This example demonstrates:
- dn.get_run_context() - Capture run context for transfer
- dn.continue_run() - Continue a run from another context
- dn.task_span() - Create task spans programmatically

Run with:
    python examples/distributed_run.py

Note: For actual distributed processing, use Ray or similar frameworks
that integrate with the Dreadnode SDK.
"""

import asyncio

import dreadnode as dn


async def main():
    # Configure SDK
    dn.configure(server="local")

    print("Distributed Run Example")
    print("=" * 50)

    # Create a tracked run
    with dn.run("distributed-demo", params={"workers": 4}):
        print("\n1. Created run...")

        # Capture run context (useful for transferring to other processes)
        run_context = dn.get_run_context()
        print(f"   Run ID: {run_context['run_id']}")
        print(f"   Run name: {run_context['run_name']}")
        print(f"   Project: {run_context['project']}")

        # Log parameters
        dn.log_params(
            computation_type="parallel",
            worker_count=4,
        )

        # Create task spans for different "workers"
        print("\n2. Running worker tasks...")

        results = []
        for worker_id in range(4):
            with dn.task_span(f"worker-{worker_id}") as task:
                # Simulate work
                await asyncio.sleep(0.05)
                result = 42 + worker_id * 10

                # Log from within the task
                dn.log_metric(f"worker_{worker_id}_result", result)
                dn.log_output(f"result", result)

                results.append(result)
                print(f"   Worker {worker_id}: result={result}")

        # Aggregate results
        total = sum(results)
        dn.log_metric("total_result", total)
        print(f"\n3. Total result: {total}")

        # The run context can be serialized and sent to other processes:
        # - Save to file: json.dumps(run_context)
        # - Pass to Ray remote: my_remote_fn.remote(run_context)
        # - Send via message queue: queue.put(run_context)
        #
        # In the remote process:
        #   dn.configure(server="local")
        #   with dn.continue_run(run_context):
        #       # Work here is linked to the parent run
        #       dn.log_metric("remote_metric", value)

    print("\n" + "=" * 50)
    print("Run complete!")
    print(f"\nRun context for remote processes:")
    print(f"  {run_context}")


if __name__ == "__main__":
    asyncio.run(main())
