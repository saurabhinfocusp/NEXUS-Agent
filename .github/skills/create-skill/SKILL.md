---
name: create-skill
description: 'Create or refine a VS Code agent SKILL.md from a conversation workflow. Use when turning a repeatable process into a workspace or personal skill, choosing its scope and primitive, drafting YAML frontmatter and procedures, validating the result, or identifying ambiguities before finalizing.'
argument-hint: '[workflow or conversation context]'
user-invocable: true
---

# Create A Skill

Turn a repeatable workflow into a discoverable, self-contained agent skill. Preserve the useful reasoning in the workflow without copying conversation-specific details.

## When To Use

- A conversation contains a repeatable implementation, debugging, review, research, or documentation process.
- The user wants a new `SKILL.md` or wants to improve an existing skill.
- The desired workflow has multiple steps, decision points, or completion checks.

## Procedure

1. Review the available conversation history and identify the concrete workflow being followed.
2. Extract three things:
   - ordered steps and the purpose of each step;
   - decision points, branching conditions, and unresolved assumptions;
   - quality criteria and checks that define completion.
3. If the workflow is unclear, ask focused questions before drafting:
   - What outcome should the skill produce?
   - Should it be workspace-scoped and team-shared, or personal and cross-workspace?
   - Should it be a short checklist or a full multi-step workflow?
4. Choose the smallest suitable customization primitive. Use a skill for an on-demand workflow with reusable procedural guidance or bundled resources; use instructions for always-on behavior, a prompt for one focused task, or a custom agent when context isolation or different tool restrictions are required.
5. Choose the installation location:
   - workspace/team-shared: `.github/skills/<skill-name>/SKILL.md`;
   - personal: `~/.copilot/skills/<skill-name>/SKILL.md`, `~/.agents/skills/<skill-name>/SKILL.md`, or `~/.claude/skills/<skill-name>/SKILL.md` according to the user's agent environment.
6. Draft `SKILL.md` with:
   - YAML frontmatter containing `name` and a keyword-rich `description`;
   - a concise purpose statement;
   - explicit triggers under `When To Use`;
   - numbered procedures with decision branches stated as conditions;
   - completion checks and references to any bundled resources using relative `./` paths.
7. Keep the skill self-contained and progressively loadable. Keep `SKILL.md` under 500 lines, avoid conversation-specific names, and move large examples or reusable assets into sibling `references/`, `scripts/`, or `assets/` directories.
8. Save the draft, then inspect the most ambiguous or weakest part. Ask one focused question about that point before treating the skill as final.
9. After the user resolves the ambiguity, update the skill and validate it:
   - `name` is 1-64 lowercase alphanumeric characters or hyphens and matches the directory name;
   - YAML is between the opening and closing `---` markers, with no tabs or malformed values;
   - `description` is present, meaningful, and contains concrete discovery keywords;
   - all referenced files exist and use relative paths;
   - the procedure and completion checks are sufficient for another agent to execute independently.
10. Report what the skill produces, give a few example prompts that should trigger it, and suggest only closely related customizations that would extend the workflow.

## Completion Criteria

A skill is ready when its scope and primitive are explicit, its frontmatter is valid, its procedure can be followed without the original conversation, and its validation checks cover both structure and behavior.

## Draft Review Question

Before finalizing, ask the user to resolve the single highest-impact ambiguity, such as the intended scope, the expected artifact, or the point where the workflow should stop and hand control back to the user.
