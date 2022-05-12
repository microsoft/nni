# Test Filter

## Checks

### Basic Check

Pipeline run should be triggerd by github pull request. If not, run following tests.

### File Changes Check

Check if any changed files match glob patterns provided by pipeline variable `nni.ci.file.globs`. If matches, run following tests.

File is compared (git diff -stat) between current commit and pr target branch.

### Pull Request Body Check

The script will try to find the task list under the heading provided by pipeline variable `nni.ci.markdown.heading` from the pull request body markdown.

If the task list is not found, continue.

If any selected options match provided index / value, run following tests.

## Pipeline Variables

### [PR body check] `nni.ci.githubPAT`

Github personal access token used to access the pr body. Could be ignored if the repo is public.

Should be a secret variable.

### [PR body check] `nni.ci.markdown.heading`

Heading used by pr body check to locate the task list.

### [PR body check] `nni.ci.markdown.optionIndex` / `nni.ci.markdown.optionValue`

Option index (start from 0) / value to be compared with selected options in the located task list.

### [File Changes Check] `nni.ci.file.globs`

Comma seperated glob patterns. See [multimatch](https://github.com/sindresorhus/multimatch#globbing-patterns)'s doc for more information.

### [Output] `skipsubsequent`

Output variable. If following tests need to be skipped, it will be set to true.

Please refer to [azure pipeline's doc](https://docs.microsoft.com/en-us/azure/devops/pipelines/process/conditions?view=azure-devops&tabs=yaml) and [sample pipeline yaml](./sample_pipelines.yml) for more information.
