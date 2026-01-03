"""Packages Example - Dataset and Model Packages.

This example demonstrates:
- dn.init_package() - Initialize a new package
- dn.push_package() - Push package to registry
- dn.pull_package() - Pull package from registry
- dn.load() - Load a package
- Dataset creation and management

Run with:
    python examples/packages.py
"""

import dreadnode as dn
from dreadnode import Dataset


def create_math_dataset():
    """Create a sample math problems dataset."""
    print("Creating math dataset...")

    # Create a new dataset
    dataset = Dataset()

    # Add training split with math problems
    train_data = [
        {"question": "What is 2 + 2?", "answer": 4, "difficulty": "easy"},
        {"question": "What is 15 - 7?", "answer": 8, "difficulty": "easy"},
        {"question": "What is 6 * 7?", "answer": 42, "difficulty": "medium"},
        {"question": "What is 144 / 12?", "answer": 12, "difficulty": "medium"},
        {"question": "What is 25 * 4 + 10?", "answer": 110, "difficulty": "hard"},
        {"question": "What is (100 - 20) / 4?", "answer": 20, "difficulty": "hard"},
    ]
    dataset.add_split("train", train_data)

    # Add test split
    test_data = [
        {"question": "What is 3 + 5?", "answer": 8, "difficulty": "easy"},
        {"question": "What is 9 * 8?", "answer": 72, "difficulty": "medium"},
        {"question": "What is 200 / 8 - 5?", "answer": 20, "difficulty": "hard"},
    ]
    dataset.add_split("test", test_data)

    return dataset


def main():
    # Configure SDK (local mode - no server needed)
    dn.configure(server="local")

    print("Packages Example")
    print("=" * 50)

    # Initialize a new dataset package
    print("\n1. Initializing package...")
    package = dn.init_package("example-math-problems", "datasets")
    print(f"   Package path: {package.path}")

    # Create and save dataset
    print("\n2. Creating dataset...")
    dataset = create_math_dataset()

    # Get split info
    train_df = dataset.to_pandas("train")
    test_df = dataset.to_pandas("test")
    print(f"   Train samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")

    # Save to package
    print("\n3. Saving dataset to package...")
    dataset.save(package.path)
    print("   Dataset saved!")

    # Push package (local only in this example)
    print("\n4. Pushing package...")
    result = dn.push_package(package.path)
    print(f"   Push result: {result}")

    # Load it back
    print("\n5. Loading dataset...")
    loaded = dn.load("example-math-problems")
    print(f"   Loaded type: {type(loaded).__name__}")

    # Access data
    print("\n6. Accessing data...")
    train_items = loaded.to_list("train")
    print(f"   Train items: {len(train_items)}")
    print(f"   First item: {train_items[0]}")

    # Use in training context
    print("\n7. Preparing for training...")
    prompts = [item["question"] for item in train_items]
    answers = [item["answer"] for item in train_items]
    print(f"   Prompts: {len(prompts)}")
    print(f"   Sample: '{prompts[0]}' -> {answers[0]}")

    print("\n" + "=" * 50)
    print("Package example complete!")
    print("\nYou can now use this dataset with dn.train():")
    print('  dn.train({"dataset": {"type": "dreadnode", "name": "example-math-problems", ...}})')


if __name__ == "__main__":
    main()
