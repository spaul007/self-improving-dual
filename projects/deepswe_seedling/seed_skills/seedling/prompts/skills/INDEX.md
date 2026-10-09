# Skill index

One line per skill: `- <name> | roles: <role>[, <role>...] | <when to use it>`.
Roles: patch, verify, baseline. Every line must have a matching `<name>.md` file in this directory.

- requirements-checklist | roles: patch, verify | at the start of your work, before editing or testing anything
- literal-contracts | roles: patch, verify | when the task names exact identifiers, messages, formats, flags or file names
- project-test-runner | roles: patch, verify, baseline | before running any test or build command in the repo
- build-before-handoff | roles: patch | before you finish, whenever you changed Go, Rust, TypeScript or other compiled code
- evidence-per-requirement | roles: verify | before you give any verdict
