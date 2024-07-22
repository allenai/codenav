import copy
import multiprocessing as mp
import sys
import time
import traceback
from queue import Empty
from typing import Any, Iterator, Sequence

from codenav.utils.eval_types import EvalInput, EvalSpec

if sys.platform.lower() == "darwin":
    mp = mp.get_context("spawn")
else:
    mp = mp.get_context("forkserver")


def _parallel_worker(task_queue: mp.Queue, result_queue: mp.Queue, eval_spec: EvalSpec):
    while True:
        try:
            eval_input: EvalInput = task_queue.get(timeout=1)
            print(f"Starting task: {eval_input.uid}")
            try:
                result = CodenavEvaluator.evaluate_input(
                    eval_input=eval_input, eval_spec=copy.deepcopy(eval_spec)
                )
            except:
                result = ("failure", eval_input, traceback.format_exc())
                result_queue.put(result)
                raise
            result_queue.put(result)
        except Empty:
            break


class CodenavEvaluator:
    def __init__(self, eval_spec: EvalSpec):
        self.eval_spec = eval_spec

    @staticmethod
    def evaluate_input(
        eval_input: EvalInput,
        eval_spec: EvalSpec,
    ) -> Any:
        episode = eval_spec.build_episode(
            eval_input=eval_input, episode_kwargs=eval_spec.episode_kwargs
        )
        interaction_output = eval_spec.run_interaction(
            episode=episode, interaction_kwargs=eval_spec.interaction_kwargs
        )
        assert interaction_output is not None
        return eval_spec.log_output(
            interaction_output=interaction_output,
            eval_input=eval_input,
            logging_kwargs=eval_spec.logging_kwargs,
        )

    def evaluate_in_sequence(self, inputs: Sequence[EvalInput]) -> Iterator[Any]:
        for input in inputs:
            yield CodenavEvaluator.evaluate_input(input, self.eval_spec)

    def evaluate_in_parallel(
        self, inputs: Sequence[EvalInput], n_procs: int
    ) -> Iterator[Any]:
        task_queue: mp.Queue[EvalInput] = mp.Queue()
        result_queue: mp.Queue[Any] = mp.Queue()

        for input in inputs:
            task_queue.put(input)

        procs = []
        for proc_idx in range(n_procs):
            p = mp.Process(
                target=_parallel_worker, args=(task_queue, result_queue, self.eval_spec)
            )
            p.start()
            procs.append(p)

        for _ in range(len(inputs)):
            yield result_queue.get()

        for proc in procs:
            proc.join(1)

    def evaluate(self, samples: Sequence[EvalInput], n_procs: int = 1) -> Iterator[Any]:
        start_time = time.time()

        if n_procs > 1:
            yield from self.evaluate_in_parallel(samples, n_procs)
        else:
            yield from self.evaluate_in_sequence(samples)

        print(f"Time taken in evaluation: {time.time() - start_time}")
