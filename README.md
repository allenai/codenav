# [CodeNav: Beyond tool-use to using real-world codebases with LLM agents 🚀](https://codenav.allenai.org/)

<div align="center">
    <img src="https://codenav.allenai.org/static/images/teaser_v3.jpg" alt="Visualization of the CodeNav agent. A user query is processed by an agent that interfaces with several environments to write code to answer the query." width="100%">
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2406.12276-red.svg)](https://arxiv.org/abs/2406.12276)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


CodeNav is an LLM agent that navigates and leverages previously unseen code repositories to solve user queries. In contrast to tool-use LLM agents that require "registration" of all relevant tools via manual descriptions within the LLM context, CodeNav automatically indexes and searches over code blocks in the target codebase, finds relevant code snippets, imports them, and uses them to iteratively generate a solution with execution feedback.

## Getting Started 🛠️

You can use CodeNav as a command line tool or programmatically as a Python module. In either case, you'll first
want to install CodeNav:
```bash
pip install git+https://github.com/allenai/codenav 
```

### CodeNav as a command line tool

After installing `codenav`, you can use it as a command line tool by running :
```bash
codenav init # Downloads/starts the Elasticsearch search index CodeNav depends to search for code snippets 
```
and then
```bash
codenav query \
  --code_dir /PATH/TO/CODEBASE/YOU/WANT/CODENAV/TO/USE \
  --playground_dir /WORKING/DIRECTORY/FOR/CODENAV/AGENT \
  --query "Query you want CodeNav to answer using the above codebase"
```
You can find other command line options by running `codenav --help`. For example, you might run something like
```bash
codenav query \
  --code_dir /PATH/TO/THIS/REPO/codenav \
  --playground_dir /PATH/TO/THIS/REPO/playground \
  --query "Write a google-style documentation string for the DoneEnv class and save it to DoneEnv.py"
```
Running the above results in the CodeNav agent saving a file `DoneEnv.py` with contents:
<details>
    <summary>Click to see DoneEnv.py contents</summary>

    class DoneEnv(CodeNavEnv):
        """
        DoneEnv is an environment class that handles the 'done' action in the CodeNav framework.

        Methods:
            check_action_validity(action: CodeNavAction) -> Tuple[bool, str]:
                Checks if the given action is valid for the 'done' action.

            step(action: CodeNavAction) -> None:
                Executes the 'done' action.
        """
        def check_action_validity(self, action: CodeNavAction) -> Tuple[bool, str]:
            """
            Checks if the given action is valid for the 'done' action.

            Args:
                action (CodeNavAction): The action to be validated.

            Returns:
                Tuple[bool, str]: A tuple containing a boolean indicating validity and an error message if invalid.
            """
            assert action.content is not None

            if action.content.strip().lower() in ["true", "false"]:
                return True, ""
            else:
                return (
                    False,
                    "When executing the done action, the content must be either 'True' or 'False'",
                )

        def step(self, action: CodeNavAction) -> None:
            """
            Executes the 'done' action.

            Args:
                action (CodeNavAction): The action to be executed.
            """
            return None
</details>

Note: the `codenav` command line tool is simply an alias for running the [codenav_run.py](codenav%2Fcodenav_run.py) so 
you can replace `codenav ...` with `python -m codenav.codenav_run ...`
or `python /path/to/codenav/codenav_run.py ...` and obtain the same results.

Here's a more detailed description of the arguments you can pass to `codenav query` or `python -m codenav.codenav_run query`:
| Argument | Type | Description |
| --- | --- | --- |
| `--code_dir` | str | The path to the codebase you want CodeNav to use. By default all files in this directory will get indexed with relative file paths. For instance, if you set `--code_dir /Users/tanmay/codebase` which contains a `computer_vision/tool.py` file then this file will be indexed with relative path `computer_vision/tools.py` |
| `--force_subdir` | str | If you wish to only index a subdirectory within the code_dir then set this to the name of the sub directory |
| `--module` | str | If you have a module installed e.g. via `pip install transformers` and you want CodeNav to use this module, you can simply set `--module transformers` instead of providing `--code_dir` |
| `--repo_description_path` | str | If you have a README file or a file with a description of the codebase you are using, you can provide the path to this file here. You may use this file to point out the libraries high-level purpose and structure of the codebase (e.g. highlight important directories, files, classes or functions) |
| `--force_reindex` | bool | Set this flag if you want to force CodeNav to reindex the codebase. Otherwise, CodeNav will reuse an existing index if it exists or create one if it doesn't |
| `--playground_dir` | str | The path specified here will work as the current directory for CodeNav's execution environment |
| `--query` | str | The query you want CodeNav to solve using the codebase |
| `--query_file` | str | If your query is long, you may want to save it to a txt file and provide the path to the text file here |
| `--max_steps` | int | The maximum number of interactions to allow between CodeNav agent and environments |


### CodeNav as a library

If you'd like to use CodeNav programmatically, you can do so by importing the `codenav` module and using the various
functions/classes we provide. To get a sense of how this is done, we provide a number of example scripts
under the [codenav_examples](codenav_examples) directory:
- [create_index.py](codenav_examples%2Fcreate_index.py): Creates an Elasticsearch index for this codebase and then uses the `RetrievalEnv` environment to search for a code snippet.
- [create_episode.py](codenav_examples%2Fcreate_episode.py): Creates an `OpenAICodeNavAgent` agent and then uses it to generate a solution for the query `"Find the DoneEnv and instantiate it"` **on this codebase** (i.e. executes a CodeNav agent on the CodeNav codebase). Be sure to run the `create_index.py` script above to generate the index before running this script.
- [create_code_env.py](codenav_examples%2Fcreate_code_env.py)): Creates a `PythonCodeEnv` object and then executes a given code string in this environemnt
- [create_prompt.py](codenav_examples%2Fcreate_prompt.py): Creates a custom prompt and instantiates and CodeNav agent with that prompt.
- [parallel_evaluation.py](codenav_examples%2Fparallel_evaluation.py): Demonstrates how to run multiple CodeNav agents in parallel. This is useful for evaluating on a dataset of queries using multiple processes. The EvalSpec abstraction also helps you organize the code a little better!

**Note** - You will still need to launch ElasticSearch server before running any of the above. To do so run
```
python -m codenav.codenav_run init
```

## Elasticsearch & Indexing Gotchas 🤔

When running CodeNav you must start an Elasticsearch index on your machine (e.g. by running `codenav init`)
and once you run a query on a given codebase, CodeNav will index that codebase exactly once.
This process means there are two things you should keep in mind:
1. You must manually shut off the Elasticsearch index once you are done with it. You can do this by running `codenav stop`.
2. If you modify/update the codebase you are asking CodeNav to use the Elasticsearch index will not automatically update and thus CodeNav will be writing code using stale information. In this case, you should add the `--force_reindex` flag when running `codenav query`, this will force CodeNav to reindex the codebase.
3. If you run CodeNav and find that it is unable to search for a file, you may want to make sure the file was indexed correctly. You can inspect all indexed files using Elasticsearch's Kibana interface at `http://localhost:5601/`. To view all the indices index by CodeNav, go to `http://localhost:5601/app/management/data/index_management`. Then click on the index you want to inspect and the click on "Discover Index" on the top-right side of the page. This will show you all the code blocks stored in this index. You can now use the UI to run queries against this index and see if the file your are looking for is present in the index and if it has the correct file path. 

## Warning ⚠️

CodeNav is a research project and may make errors. As CodeNav can potentially execute ANY code
it wants, it is not suitable for security sensitive applications. We strongly recommend
that you run CodeNav in a sandboxed environment where data loss or security breaches are not a concern.

## Authors ✍️
- [Tanmay Gupta](https://tanmaygupta.info/)
- [Luca Weihs](https://lucaweihs.github.io/)
- [Aniruddha Kembhavi](https://anikem.github.io/)

## License 📄
This project is licensed under the Apache 2.0 License.

## Citation
```bibtex
@misc{gupta2024codenavtooluseusingrealworld,
  title={CodeNav: Beyond tool-use to using real-world codebases with LLM agents}, 
  author={Tanmay Gupta and Luca Weihs and Aniruddha Kembhavi},
  year={2024},
  eprint={2406.12276},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2406.12276}, 
}
```
