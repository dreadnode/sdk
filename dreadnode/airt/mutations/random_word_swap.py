from rigging.transform import Transform


def random_word_swap(
    words_to_swap: list[str],
    swap_with: list[str],
    swap_probability: float = 0.5,
) -> Transform:
    """
    Create a transform that randomly swaps words in a text with specified alternatives.

    Args:
        words_to_swap (list[str]): List of words to be swapped.
        swap_with (list[str]): List of words to swap with.
        swap_probability (float): Probability of swapping each word.

    Returns:
        Transform: A transform that applies the word swap.
    """

    def transform(text: str) -> str:
        import random

        words = text.split()
        for i, word in enumerate(words):
            if word in words_to_swap and random.random() < swap_probability:
                replacement = random.choice(swap_with)
                words[i] = replacement
        return " ".join(words)

    return transform
