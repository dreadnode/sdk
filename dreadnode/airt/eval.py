from rigging import ChatPipeline


class EvalPipeline(ChatPipeline):
    """
    A pipeline for evaluating the performance of a model where the outputs are known.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_results = []

    def evaluate(self, data):
        """
        Evaluate the model on the provided data.
        """
        # Implement evaluation logic here
