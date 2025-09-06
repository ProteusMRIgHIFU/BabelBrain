# Goal
Review all Python files in this repository. Scan every function and method definition. Identify all function arguments that do not comply with PEP-8 naming conventions.

# What to look for
- Argument names should be in snake_case.
- Names should be lowercase, descriptive, and avoid single letters except for common cases like `x`, `y`, or `z` in mathematical contexts.
- Avoid camelCase, PascalCase, or ALLCAPS for arguments.

# What to do
1. Generate a report listing:
   - The file name and line number of the function.
   - The original function signature.
   - The arguments that need renaming.
   - Suggested PEP-8 compliant replacements.

2. Propose code changes for each file:
   - Rename the arguments in the function definition.
   - Rename all references to those arguments inside the function body.
   - Do not change external calls automatically — just flag them if detected, so I can confirm.

# Output format
- Start with a summary table of all issues found.
- Then provide diffs (in `git diff` format) or updated code blocks showing suggested changes.
