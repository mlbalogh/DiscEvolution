# Merging Tobin's improvements into DiscEvolution/master

**Strategy:** keep our `master` as the base (it has the student `StartHere` material,
the `tests/` suite, and our history) and port Tobin's core `DiscEvolution/` improvements
onto a throwaway branch, validating with the tests after each step. `master` is never
touched until the final PR.

Tobin's repo has **unrelated history** (no common ancestor), so we do NOT `git merge` it.
We diff the trees and adopt changes file by file.

---

## Phase 0 — One-time setup

```
# start from a clean, current master
git checkout master
git status                     # must be clean
git fetch origin && git pull --ff-only
git fetch tobin                # tobin remote already added

# create the working branch and back it up
git checkout -b merge-tobin
git push -u origin merge-tobin

# establish a green baseline BEFORE changing anything
python -m pytest tests/        # (use whatever you normally run) -> record pass/fail

# add this tracker and commit
git add MERGE_PROGRESS.md
git commit -m "start Tobin merge: progress tracker + baseline"
git push
```

If the baseline tests don't all pass on untouched master, note which — so later you can
tell a pre-existing failure from one you introduced.

---

## Phase 1 — Port the core library, one file per (small) commit

Work easiest → hardest so the workflow is proven before the big files. For each file:

1. See ONLY the real changes (strip formatting noise):
   ```
   git diff -w --ignore-blank-lines master tobin/main -- DiscEvolution/<file>
   ```
2. Choose the approach for that file:
   - **Mostly `+` lines (Tobin only added):** safe to take his file wholesale —
     `git checkout tobin/main -- DiscEvolution/<file>` — then re-read the diff to confirm.
   - **Meaningful `-` lines (content on master that Tobin lacks = our own edits):**
     hand-apply his real changes into our file in the editor, keeping our lines.
     The big two (`dust.py`, `planet_formation.py`) are this case — we have edits there
     (e.g. the `# MLB - restored` line), so hand-merge, and go **function by function**,
     one commit each, not the whole file at once.
3. Run the tests: `python -m pytest tests/`
4. Commit when acceptable:
   ```
   git add DiscEvolution/<file>          # or the specific hunk
   git commit -m "port Tobin's changes to <file>"
   ```
5. Tick the checklist below, commit that, and `git push`.

### Checklist (easiest first)

- [ ] `constants.py`          (31, mixed)
- [ ] `disc.py`               (5, additions)
- [ ] `star.py`               (13, mixed)
- [ ] `viscous_evolution.py`  (22, mixed)
- [ ] `dust.py`               (873 — break into function-by-function commits)
- [ ] `planet_formation.py`   (826 — break into function-by-function commits)

---

## Phase 2 — Adopt Tobin's new standalone files (trivial, no conflicts)

These don't exist on master, so just check them out. Pick the ones you want:

```
git checkout tobin/main -- notebooks/            # his new analysis notebooks
git checkout tobin/main -- scripts/run_model.py scripts/run_model_stream.py
git add -A && git commit -m "adopt Tobin's notebooks and run scripts"
git push
```

- [ ] Decide which of his notebooks/scripts to bring in
- [ ] Committed

(Skip his `.log`, `completed.txt`, `backups/` etc. — run artifacts you don't want.)

---

## Phase 3 — Finish

```
git diff master...merge-tobin --stat     # review the whole branch's net change
python -m pytest tests/                  # full green run
```

- [ ] Full diff reviewed
- [ ] All tests green
- [ ] Open PR `merge-tobin -> master` on GitHub, review, merge
- [ ] Tell students to sync their forks from master (`git fetch upstream && git merge upstream/master`)
- [ ] (optional) delete `merge-tobin`; `git remote remove tobin`

---

## Resuming after a break

```
git checkout merge-tobin
git pull
cat MERGE_PROGRESS.md                     # what's done / next
git log --oneline master..merge-tobin     # commits so far on this branch
```

## Undo cheatsheet (master is safe regardless)

- Discard uncommitted edits to one file:   `git checkout -- DiscEvolution/<file>`
- Discard ALL uncommitted work (keep commits):  `git reset --hard HEAD`
- Undo the last commit, keep the edits staged:   `git reset --soft HEAD~1`
- Reverse an already-committed step:   `git revert <commit>`
