"""Package for all CLI commands in the project.

Commands should be placed in this directory as modules. Each module should
define a click command (function decorated with @click.command) whose name
matches the name of the module. Then that command can be executed using that
name (with underscores replaced with hyphens) on the command line.

For example if train_model.py in this directory contains a click command
names train_model, you can then run this script using:

$ python -m deepsurfer-train train-model argument1 -o option1

Or is the package is currently installed in your environment:
$ deepsurfer-train train-model argument1 -o option1

"""
