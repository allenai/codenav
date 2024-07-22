from typing import Sequence

from codenav.interaction.messages import UserQueryToAgent

PATHS = """\
Use the code base located at `{CODE_DIR}` to solve this query. Your current directory is `{WORKING_DIR}`.
"""

ADDED_TO_PATH = """\
The code base path has either been installed via pip or has been already been added to the system path via
```
import sys
sys.path.extend({ADDED_PATH})
```
"""

# Import instructions might be different in case of an installed library rather than a local code repo. For now, we assume that the code repo is local and not installed.
IMPORT_INSTRUCTIONS = """\
If the import path in retrieved code block says `testing/dir_name/file_name.py` and you want to import variable, function or class called `obj` from this file, then import using `from testing.dir_name.file_name import obj`.
"""


def create_user_query_message(
    user_query_str: str,
    code_dir: str,
    working_dir: str,
    added_paths: Sequence[str],
):
    return UserQueryToAgent(
        message=f"USER QUERY: {user_query_str}"
        + (f"\n" f"\n{PATHS}" f"\n{IMPORT_INSTRUCTIONS}" f"\n{ADDED_TO_PATH}").format(
            CODE_DIR=code_dir, WORKING_DIR=working_dir, ADDED_PATH=list(added_paths)
        ),
    )
