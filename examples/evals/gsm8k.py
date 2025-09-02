# import rigging as rg
# from datasets import load_dataset

# import dreadnode as dn
# from dreadnode import Eval


# class Answer(rg.Model):
#     reasoning: str = rg.element(description="Your reasoning.")
#     final_answer: float = rg.element(description="Single float value.")


# def prepare_gsm8k(row: dict) -> dict:
#     reasoning, answer = row["answer"].split("####")
#     return {
#         "question": row["question"],
#         "reasoning": reasoning.strip(),
#         "answer": float(answer.strip()),
#     }


# gsm8k_dataset = (
#     load_dataset("gsm8k", "main", split="test").select(range(10)).map(prepare_gsm8k).to_list()
# )

# # Define the evaluation


# class MyEval(Eval):
#     name = "GSM8K"
#     dataset = gsm8k_dataset
#     parameters = {"model": ["gpt-4o-mini", "claude-3-5-haiku-latest"]}
#     scorers = [
#         dn.scorers.equals(dn.DatasetField("answer")),
#         dn.scorers.similarity(dn.DatasetField("reasoning")),
#     ]
#     assert_scores = ["correct"]
#     concurrency = 3


# # Run the evaluation

# MyEval().console()
