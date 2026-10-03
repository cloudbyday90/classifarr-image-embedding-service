# Classifarr native-validation AI skill

Assessment: 2026-10-03. The project-local [skill](../.agents/skills/classifarr-native-validation/SKILL.md) routes OS-boundary validation and CI maintenance. It complements the resource-contract and capacity-calibration skills without turning ordinary dependency or release work into platform validation.

## Design and recommendation

[Official skill authoring guidance](https://learn.chatgpt.com/docs/build-skills?translationFallback=ja-JP), discovered through search and checked against local skill-creator/OpenAI documentation, describes versioned instructions with optional resources and project `.agents/skills` discovery. Use concise instructions, one conditional command/boundary reference and quoted UI metadata. Reuse repository executables rather than duplicating scripts in the skill.

Pros: clear claim-to-test mapping, complete target graph checks, mandatory execution rather than collected/skipped tests, and explicit separation of local/hosted/model evidence. Cons: project paths and suite contracts need maintenance; instructions and manual scenario checks cannot establish autonomous model reliability.

Final stack: discriminating discovery metadata; resource-claim routing; fresh exact native environment; real resource fixtures; skip/mandatory outcome gate; sanitized evidence and limitations. Keep authority from the user/session; the skill adds no permission flow, deployment or upstream PR merge.

## Evaluation and outcome

Skill-creator's validator passes and local reference links resolve. Six practical scenarios were manually evaluated against observable results: (1) both native runs execute DACL rotation, (2) gate unit controls reject junction skips, (3) actual Windows Proactor transport runs accompany Linux results, (4) target/extra-inventory rejection controls pass, (5) discovery metadata excludes model capacity and the reference routes it separately, (6) outcome documentation distinguishes local Windows 11/older Python patches from unexecuted hosted Server 2025/latest patches. The fixed native runner passes 102 cases on each interpreter and its target/gate/audit policy controls pass 19 cases.

The skill has three files: concise instructions, one conditional reference and UI metadata. No duplicated executables, new application dependencies or authority are introduced. This is manual scenario evaluation supported by actual scripts/tests, not an independent agent benchmark. See the [native design and outcome](windows-native-validation.md) for all platform limits.
