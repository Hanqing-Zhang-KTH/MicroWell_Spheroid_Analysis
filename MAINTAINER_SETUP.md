# Maintainer Setup (Owner Checklist)

This file covers one-time GitHub settings for owner control.

## 1) Make Repository Public

1. Open repository in GitHub.
2. Go to `Settings` -> `General`.
3. Scroll to `Danger Zone`.
4. Click `Change repository visibility`.
5. Select `Public` and confirm.

## 2) Keep Collaborator Access Restricted

1. Go to `Settings` -> `Collaborators and teams`.
2. Keep no collaborators unless needed.
3. If needed, prefer lowest permission first (`Read` or `Triage`).
4. Avoid granting `Write`/`Maintain`/`Admin` unless fully trusted.

## 3) Protect `main` Branch

1. Go to `Settings` -> `Branches`.
2. Add a branch protection rule for `main`.
3. Enable:
   - `Require a pull request before merging`
   - `Require approvals` (recommended: 1)
   - `Dismiss stale pull request approvals when new commits are pushed`
   - `Block force pushes`
   - `Do not allow deletions`
4. Optional strict mode:
   - `Include administrators` (owner also follows PR flow)

## 4) Optional Ruleset (If Available)

If your GitHub plan shows `Rulesets`:

1. Go to `Settings` -> `Rules` -> `Rulesets`.
2. Create a ruleset targeting `main`.
3. Enforce PR-only merges and block direct pushes.

## 5) Recommended Merge Strategy

In `Settings` -> `General` -> `Pull Requests`:

- Enable one merge method you prefer (recommended: `Squash`).
- Optionally disable direct merge methods you do not use.

## Verification

After setup:

1. Try to push directly to `main` from a local test branch merge scenario.
2. Confirm GitHub blocks direct push and requires PR.
3. Confirm PR requires review before merge.
