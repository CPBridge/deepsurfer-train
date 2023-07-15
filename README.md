# `deepsurfer-train`

Code for training and evaluating Deepsurfer models.

This README describes the basic setup and use of the project codebase. For
information about the specific processes implemented in this codebase, refer
to the [docs](docs/) directory.

### Development Installation

The project's dependencies are designed to be managed with the `poetry` tool.
You will need to have poetry installed on your system. See
[here](https://python-poetry.org/docs/).

##### On Martinos Machines

You should use poetry and pyenv to manage your environment for this project on
Martinos servers. See
[this page](https://github.com/QTIM-Lab/qtim-standards/blob/main/environment_setup.md)
for setup instructions for these tools.

There is a python environment called `deepsurfer-train` in the
shared qtim pyenv installation allocated for this project. The easiest way to
get started is just to use this shared environment by running:

```
pyenv activate deepsurfer-train
```

It may also be convenient to tell pyenv to activate the `deepsurfer-train` environment whenever your shell is currently in
the local repository, which should be cloned into your home directory. You can
do this by running this from the repository root:

```
pyenv local deepsurfer-train
```

Since this environment is shared between multiple users who may be running
different versions of this repository from different local clones of it, the
python package defined in this repository should **not** be installed in the
pyenv environment (even though it is set up to be an installable package).

Since this repository is not in the installed environment, you need to make the
python module implemented in this package (called
`deepsurfer_train`) findable by
Python, separately from the environment with dependencies. You can either
simply run all commands from the root directory of the repository and/or add
that location (e.g. `/home/user/repos/deepsurfer-train`) to your `PYTHONPATH`
environment variable for the current shell session:

```
export PYTHONPATH=/home/user/repos/deepsurfer-train:${PYTHONPATH}
```

##### On Other Machines (e.g. Laptop/Desktop)

When working on other machines you need to create a virtualenv for this project
using your preferred tool (virtualenv, pyenv, poetry). Then you can use
`poetry` to create the environment:

```
poetry install
```

When you are not sharing your python environment with others, it may be easiest
to install the package in editable mode, rather than setting the `PYTHONPATH`
environment variable.

```
pip install -e .
```

### Basic Usage Installation

If you just need to run functionality implemented in the repo without editing
the code at all, you can just install the python package using pip, which will
also handle all dependencies. You can do this either from a clone of the code:

```
pip install /path/to/cloned/repo
```

or from directly from GitHub:

```
pip install git+https://github.com/CPBridge/deepsurfer-train
```

It is still recommended to do this in a virtual environment.

### Managing Dependencies

Manage all dependencies using poetry (either running `poetry add <package>` or
editing `pyproject.toml` yourself), then running `poetry lock` to regenerate
the lock file, committing your changes to the repository, and then using
`poetry install` to actually reflect these changes in the environment.

You should not just install packages using `pip`.

### CLI Usage

The Python package provides a command line interface (CLI) when executed using
the `python -m deepsurfer_train` command, which should be
available whenever the environment that the package is on findable (either
because it is installed or added to the `PYTHONPATH`). There are various
sub-commands, each for a different process associated with the project, e.g.
training, testing, preprocessing. Each sub-command corresponds to a Python
module in the `deepsurfer_train.commands` package.

To see a list of available sub-commands, use:

```bash
python -m deepsurfer_train --help
```

To see the documentation of an individual command, e.g. `train` use:

```bash
python -m deepsurfer_train train --help
```

To add new processes to the project, add a new Python source file within the
`deepsurfer_train/commands` directory and define a
function within it with the same name as the Python file and decorated with
`@click.command()`. This will create a new CLI sub-command with the same name
as the file and function, except with any underscores replaced with hyphens.


### Pre-commit

The repository is set up to use `pre-commit`, which will automatically check
the codebase for potential problems each time you make a commit. To ensure
that these are activated, run this once when you first clone the repository:

```
pre-commit install
```

This should be all you need to do, after this is done, various checks and
processes will be run on your code automatically before you make a commit.
Some of these processes find errors but do not fix them, others actually find
and fix problems automatically for you.

The commit will fail if any issues are found, or if any files are automatically
changed by the pre-commit hooks. Read the pre-commit output to understand the
errors that it found and the changes that were made. If changes were made
automatically, all you need to do is `git add` the changed files and re-try
the commit. If pre-commit found errors that it cannot fix, you should fix the
errors yourself before adding and committing the changed files again.

Please make use of pre-commit to help avoid errors in your code, and make it
well-organized and readable for others. It can dramatically improve the
quality of your code.

See [the pre-commit](https://pre-commit.com/) website for further guidance on
using the tool.

### Slurm Jobs

Batch scripts to run jobs associated with this project are found in the `slurm`
directory for use on the MLSC cluster. To run these jobs, you should clone a
copy of this repository into your Martinos home directory and activate the
project's environment (as above) *BEFORE* submitting the job.

### Docker Environment (Optional)

You should be able to use this codebase in a correctly set-up virtualenv
without needing a docker environment. However, there are also a set of files
to help you build and run a docker container set up to perform all tasks
associated with the project, if you need it.

The `build_docker.sh` script will build the container and tag it with the
name `deepsurfer-train`.

To start an interactive session within the project Docker container, use the
`run_docker.sh` script. This will mount the cluster storage on the host system
into the container (at the same path). Furthermore, your local copy of the
project repository on the host system will be bind-mounted into the container
at `/deepsurfer_train`.

### Singularity (Optional)

You can build a singularity container for the project using the
`build_singularity.sh` script in the `singularity` directory. This container
will contain a suitable version of Python and CUDA, along with all the
Python packages dependencies for the project defined in `pyproject.toml`. You
can run an interactive singularity container using the `run_singularity.sh`
script in the same directory.

### Slurm Jobs with Docker (Optional)

The project Docker image is built with a special entrypoint script for use on
the Slurm computing clusters. This script is placed into the container at
`/slurm_entrypoint.sh`. The script clones the project source code from the
Gitlab repository, installs the Python package within it, and then executes a
command from the package as specified by its command line arguments.

### Data

The data for this project is stored in the directory
`/autofs/cluster/freesurfer/test_data` and other project files at
`/autofs/cluster/qtim/projects/deepsurfer`.
