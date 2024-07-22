RESTART_PROMPT = """\
This task is about to be handed off to another AI agent. The agent will be given the same prompt you were and will be expected to continue your work in the same Python environment. The other agent will only be shown your single next response and will not have access to your previous thoughts, keywords, code, execution results, or retrieved code blocks. With this in mind, please respond in the format:

<thought>
A summary of everything you have done and learned so far relevant to the task. You should be concise but thorough.
</thought>
<content>
Code that, if run, would result in the Python environment being in its current state. This should include all necessary imports, definitions, and any other necessary setup. Ideally, this code should not produce any errors. Add comments when necessary explaining the code and describing the state of the environment. In this code do not define any variables/functions or import any libraries that are not available in the current environment state. For reference, the current variables defined in the environment are:\n{current_env_var_names} 
</content>
"""

PICKUP_PROMPT = """\
You are picking up where another AI agent left off. The previous agent has provided a summary of their work and a code snippet that, if run, would result in the Python environment being in its current state.
"""
