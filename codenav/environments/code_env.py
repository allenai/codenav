import copy
import math
import os
import re
import signal
import traceback
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import pandas as pd

from codenav.environments.abstractions import CodeNavEnv
from codenav.interaction.messages import CodeNavAction, ExecutionResult
from codenav.utils.linting_and_type_checking_utils import (
    LintingError,
    TypeCheckingError,
    black_format_code,
    get_linting_errors,
    get_type_checking_errors,
)

DISPLAY_UPDATED_VARS_IN_LOGS = True


def is_double_underscore_format(s):
    """Returns True if the string is in the format __.*__"""
    pattern = r"^__.*__$"
    return bool(re.match(pattern, s))


def softcopy_vars(vars: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a copy of the given vars. If the dict contains non-pickle-able objects (like
    modules), the original copy of such objects are returned."""
    softcopy = dict()
    for var_name, var in vars.items():
        try:
            softcopy[var_name] = copy.deepcopy(var)
        except Exception:
            softcopy[var_name] = var

    return softcopy


class CodeEnvTimeoutError(TimeoutError):
    pass


def exec_with_informative_errors(
    code_str: str,
    global_vars: Dict[str, Any],
    timeout: float,
    locals: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Executes a given code string within the specified global variables context and provides informative error messages.

    This function attempts to execute a given string of code (`code_str`) within the context of the provided global
    variables dictionary (`global_vars`). If the execution is successful, it returns a tuple with `True` and `None`,
    indicating success without errors. In the case of an execution error, it catches the exception, formats the error
    message to be more informative, including the file name, line number, and the offending line of code, and returns
    `False` along with the enhanced error message.

    Parameters:
        code_str (str): The string of code to be executed.
        global_vars (Dict[str, Any]): A dictionary of global variables within which the code string will be executed.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating the success or failure of the execution,
            and an optional string with an error message. The error message is `None` if execution is successful, or
            contains details of the exception if an error occurs.

    Raises:
        Exception: Captures and enhances any exception raised during the execution of `code_str`,
            without explicitly re-raising it.
    """
    try:
        # Define the function to be called on timeout
        def timeout_handler(signum, frame):
            raise CodeEnvTimeoutError(
                f"Execution timed out after {timeout} seconds. Please ensure your code runs within the time limit."
            )

        try:
            # Set the signal handler (so things time out) and set an alarm for timeout seconds
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(math.ceil(max(timeout, 0)))

            # Exec the code
            if locals is not None:
                exec(code_str, global_vars, locals)
            else:
                exec(code_str, global_vars)
        finally:
            # Cancel the alarm and reset the signal handler to the default
            signal.alarm(0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)

        return True, None

    except Exception as e:
        formatted_exc = traceback.format_exc()

        # Extract traceback details
        tb_info = traceback.extract_tb(e.__traceback__)

        # Find the last entry in the traceback (where the error occurred in the exec)
        last_call = tb_info[-1]

        # Extract line number and file name (it will be "<string>" for exec)
        line_number = last_call.lineno
        file_name = last_call.filename
        assert line_number is not None

        if file_name.strip() != "<string>":
            return False, formatted_exc

        # Attempt to get the offending line from code_str
        try:
            offending_line = code_str.split("\n")[line_number - 1].strip()
        except IndexError:
            return False, formatted_exc

        # Enhance and re-raise the error message
        enhanced_message = (
            f'{str(e)} (File "{file_name}", line {line_number}, in {last_call.name})'
            f"\n -> Error occurred when executing: {offending_line}"
        )

        return False, enhanced_message


class PythonCodeEnv(CodeNavEnv):
    """Python code environment that can execute Python sequences of code."""

    def __init__(
        self,
        code_dir: Optional[str] = None,
        init_global_vars: Optional[Dict[str, Any]] = None,
        sys_paths: Optional[Sequence[str]] = None,
        working_dir: Optional[str] = None,
        enable_black_formatting: bool = False,
        enable_linting: bool = False,
        enable_type_checking: bool = True,
        max_execution_time: float = 60,
    ):
        self.enable_black_formatting = enable_black_formatting
        self.enable_linting = enable_linting
        self.enable_type_checking = enable_type_checking

        # init_global_vars will be used to reset the environment if needed
        self.init_global_vars = init_global_vars
        self.code_dir = code_dir
        self.sys_paths = sys_paths if sys_paths is not None else []
        self.working_dir = os.path.abspath(
            working_dir if working_dir is not None else "."
        )
        self.max_execution_time = max_execution_time

        # global_vars will be updated with every step() call
        self.global_vars: Dict[str, Any] = {}

        # exec_trace is a list of ExecutionResult objects from all step calls so far
        # When step is called with overwrite_last=True, the last item in exec_trace
        # is replaced with the new ExecutionResult
        self.exec_trace: List[ExecutionResult] = []

        self._last_working_dir = self.working_dir
        self.reset(restore_to="init")

    def reset(self, restore_to: Literal["init", "blank"] = "init"):
        """Reset the environment to the initial or blank state."""

        self.exec_trace = []

        if restore_to == "init":
            self._last_working_dir = self.working_dir

            # global_vars will be updated with every step() call
            self.global_vars = (
                softcopy_vars(self.init_global_vars)
                if self.init_global_vars is not None
                else dict()
            )

            if self.sys_paths is not None:
                self.append_sys_paths(self.sys_paths)

        elif restore_to == "blank":
            raise NotImplementedError(
                "This doesn't really make sense, what should we set self.working_dir to?"
                " There isn't really a default starting directory unless we want to "
                " use `~` which seems dangerous."
            )

        else:
            raise ValueError(f"Invalid value for restore: {restore_to}")

    def check_action_validity(self, action: CodeNavAction) -> Tuple[bool, str]:
        return True, ""

    def append_sys_paths(self, sys_paths: Sequence[str]):
        """Append sys.path with the given paths."""
        self.step(f"import sys\nsys.path.extend({sys_paths})")

    def run_black_and_linting_and_mypy(
        self, code_str: str
    ) -> Tuple[str, List[LintingError], List[TypeCheckingError]]:
        """Format the code and run linting and type checking."""
        if self.enable_black_formatting:
            code_str = black_format_code(code_str)[0]

        # code_history is "" if there are no previous steps
        code_history = "\n".join([res.code_str for res in self.exec_trace])
        if code_history != "":
            code_in_context = "\n".join([code_history, code_str])
            history_len = len(code_history.split("\n"))
        else:
            code_in_context = code_str
            history_len = 0

        new_linting_errors: List[LintingError] = []
        if self.enable_linting:
            linting_errors = get_linting_errors(code_in_context)
            new_linting_errors = [
                lint_err
                for lint_err in linting_errors
                if lint_err.line_num > history_len
            ]
            for lint_err in new_linting_errors:
                lint_err.line_num = lint_err.line_num - history_len

        new_type_checking_errors: List[TypeCheckingError] = []
        if self.enable_type_checking:
            type_checking_errors = get_type_checking_errors(code_in_context)
            new_type_checking_errors = [
                type_err
                for type_err in type_checking_errors
                if type_err.line_num > history_len
            ]
            for type_err in new_type_checking_errors:
                type_err.line_num = type_err.line_num - history_len

        return code_str, new_linting_errors, new_type_checking_errors

    def step_append(self, code_str: str) -> ExecutionResult:
        """
        Execute the given code string as if appending the code to the end of the
        existing code sequence.
        """
        # format and analyze code_str
        (
            code_str,
            linting_errors,
            type_checking_errors,
        ) = self.run_black_and_linting_and_mypy(code_str)

        # Temporarily change the working directory
        saved_working_dir = os.getcwd()
        os.chdir(self._last_working_dir)

        # execute code_str
        var_name_to_repr = {k: str(v) for k, v in self.global_vars.items()}
        stdout: StringIO = StringIO()
        with redirect_stdout(stdout):  # type: ignore
            success, exec_error = exec_with_informative_errors(
                code_str=code_str,
                global_vars=self.global_vars,
                timeout=self.max_execution_time,
            )
        stdout_str = stdout.getvalue()

        # A variable is considered "updated" if its str representation changed (or it was just defined)
        updated_vars: Dict[str, Any] = {
            k: v
            for k, v in self.global_vars.items()
            if (
                k in code_str
                or (k not in var_name_to_repr and not is_double_underscore_format(k))
            )
            and str(v) != var_name_to_repr.get(k, "")
        }
        exec_result = ExecutionResult(
            code_str=code_str,
            stdout=stdout_str,
            updated_vars=updated_vars,
            exec_error=exec_error,
            linting_errors=linting_errors,
            type_checking_errors=type_checking_errors,
        )

        # Reset working directory
        self._last_working_dir = os.getcwd()
        os.chdir(saved_working_dir)

        # update trace
        self.exec_trace.append(exec_result)

        return exec_result

    def step(self, action: Union[str, CodeNavAction]) -> ExecutionResult:
        """
        Execute the given code string. If overwrite_last is True, the last result is
        removed from the execution trace. If reset_and_run_all is True, the environment
        is reset to the initial state and all previous steps are re-executed before
        executing the given code string.
        """
        if isinstance(action, str):
            return self.step_append(action)
        elif isinstance(action, CodeNavAction):
            assert action.content is not None
            return self.step_append(action.content)
        else:
            raise ValueError(f"Invalid type for action: {type(action)}")

    def step_sequence(
        self,
        code_seq: Sequence[str],
    ) -> List[ExecutionResult]:
        for code_str in code_seq:
            _ = self.step(code_str)
        seq_len = len(code_seq)
        return self.exec_trace[-seq_len:]

    def summary_str(self):
        summary_str = ""
        for i, res in enumerate(self.exec_trace):
            res_str = res.format(
                include_code=True, display_updated_vars=DISPLAY_UPDATED_VARS_IN_LOGS
            )
            summary_str += f"*Step {i+1}*\n{res_str}\n"

        return summary_str

    def tabulate(self) -> pd.DataFrame:
        records = []
        for res in self.exec_trace:
            records.append(
                dict(
                    code=res.code_str,
                    output=res.format(
                        include_code=False,
                        display_updated_vars=DISPLAY_UPDATED_VARS_IN_LOGS,
                    ),
                )
            )

        return pd.DataFrame.from_records(records)
