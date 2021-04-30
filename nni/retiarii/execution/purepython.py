from .base import BaseExecutionEngine


class PurePythonExecutinoEngine(BaseExecutionEngine):
    def submit_models(self, *models):
        pass

    @classmethod
    def trial_execute_graph(cls) -> None:
        pass
