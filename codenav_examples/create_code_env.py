import os

from codenav.environments.code_env import PythonCodeEnv

project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(f"Project directory: {project_dir}")
env = PythonCodeEnv(
    sys_paths=[project_dir],
    working_dir=os.path.join(project_dir, "playground"),
    enable_type_checking=True,
)

exec_output1 = env.step("from codenav.interaction.messages import ACTION_TYPES")
print(exec_output1.format(include_code=True, display_updated_vars=True))

exec_output2 = env.step("print(ACTION_TYPES)")
print(exec_output2.format(include_code=True, display_updated_vars=True))
