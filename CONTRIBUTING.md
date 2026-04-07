# Contributing Guide

Thank you for using and improving this project.

## Core Rule

The `main` branch is treated as protected and release-ready.  
All changes should go through a pull request (PR), including maintainer changes.

## Standard Workflow

1. Sync local repository:
   - `git checkout main`
   - `git pull origin main`
2. Create a feature branch:
   - `git checkout -b feature/<short-description>`
3. Make and test changes.
4. Commit with clear messages.
5. Push branch:
   - `git push -u origin feature/<short-description>`
6. Open a PR into `main`.
7. Complete PR checklist and request review.
8. Merge only after approval and green checks.

## Push / Pull Basics

- `git push`: upload your local commits to GitHub.
- `git pull`: download remote updates to your local branch.
- Always pull latest `main` before starting a new feature branch.

## Branch Naming

- `feature/<name>` for new functionality
- `fix/<name>` for bug fixes
- `docs/<name>` for documentation-only changes
- `chore/<name>` for maintenance changes

## Pull Request Expectations

- Explain what changed and why.
- Include a basic test plan.
- Keep PR focused and small where possible.
- Keep attribution and license files intact.

## Licensing and Attribution

This project is GPL-3.0 licensed.  
If you redistribute modified versions, preserve attribution and clearly mark changes.
