# Notebooks

This is a location to store jupyter notebooks associated with the project.
Note that:

1. Notebooks should only be used for tasks such analysis, visualization, and
   experimentation. They should not be used to run any tasks required to complete
   the project, for example:
    - Training models
    - Generating results
    - Preprocessing data
    - Postprocessing data

   These tasks should only be performed using version controlled Python code with
   a defined CLI entrypoint.

2. The output of cells should be cleared before adding them to version control
   This helps prevent PHI or large image files getting added to gitlab. The
   project share is the correct location to store analysis results or
   visualizations that you want to keep. The project's pre-commit hooks
   should automatically clear cell output before committing changes to the
   repo.
