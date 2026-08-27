# Getting Started with DiscEvolution

Welcome! This guide gets you from "nothing installed" to "running your own
protoplanetary-disc simulation and understanding what it did." It focuses on
the student-facing workflow in `example/StartHere/`: `run_model_student.py`
for a single run, and `run_popsynth_student.sh` for a batch of them.

If you get stuck anywhere in here, that's normal — ask in the group chat or
ask your editor's AI assistant (see [Section 3](#3-vs-code-ai-assistants-and-the-debugger))
to explain the error before you spend more than ~15 minutes stuck on
something that looks like plumbing rather than physics.

**Contents**
1. [Get the code](#1-get-the-code-fork--clone)
2. [Set up Python](#2-set-up-a-python-environment)
3. [VS Code, AI assistants, and the debugger](#3-vs-code-ai-assistants-and-the-debugger)
4. [Daily git workflow](#4-daily-git-workflow)
5. [Run a single model](#5-run-a-single-model)
6. [Code architecture & units](#6-code-architecture--units)
7. [Batch runs (parameter sweeps)](#7-batch-runs-parameter-sweeps)
8. [Running fully in the background](#8-running-fully-in-the-background)
9. [Where to go next](#9-where-to-go-next)

---

## 1. Get the code (fork + clone)

1. Go to https://github.com/mlbalogh/DiscEvolution and click **Fork** (top
   right). This gives you your own copy on GitHub, e.g.
   `github.com/<your-username>/DiscEvolution`.
2. Clone *your fork* to your laptop:
   ```bash
   git clone https://github.com/<your-username>/DiscEvolution.git
   cd DiscEvolution
   ```
3. Add the shared repo as a second remote called `upstream`, so you can pull
   in updates other people make later:
   ```bash
   git remote add upstream https://github.com/mlbalogh/DiscEvolution.git
   git remote -v   # should show "origin" (your fork) and "upstream"
   ```

**Windows users:** install [Git for Windows](https://gitforwindows.org/) —
it bundles **Git Bash**, a terminal that understands the `bash` scripts in
this repo (`run_popsynth_student.sh` etc.). Use Git Bash for everything in
this guide unless noted otherwise; plain Command Prompt / PowerShell can't
run these scripts directly. If you find yourself fighting the terminal a
lot, consider installing **WSL2** instead (see [Section 8](#8-running-fully-in-the-background))
— it gives you a real Linux environment and everything below just works
exactly as written for Mac users.

**Mac users:** you already have a Unix terminal (Terminal.app or iTerm2)
and probably already have `git`; if not, `xcode-select --install` gets you
a minimal one, or install Git for Mac / use Homebrew (`brew install git`).

---

## 2. Set up a Python environment

Use a **virtual environment** so this project's packages don't collide with
anything else on your machine. You need Python 3.9+.

**Mac / Linux / WSL / Git Bash:**
```bash
cd DiscEvolution
python3 -m venv .venv
source .venv/bin/activate          # do this every time you start a new terminal
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                   # installs DiscEvolution itself, editable
pip install jupyter ipykernel      # only needed for the analysis notebooks
```

**Windows (PowerShell, if not using Git Bash/WSL):**
```powershell
cd DiscEvolution
python -m venv .venv
.venv\Scripts\Activate.ps1         # do this every time you start a new terminal
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install jupyter ipykernel
```

You'll know it worked if `python -c "import DiscEvolution; import h5py"`
runs with no errors. Your terminal prompt should show `(.venv)` while the
environment is active — if you close and reopen the terminal, you need to
run the `activate` line again (but not the `pip install` lines).

---

## 3. VS Code, AI assistants, and the debugger

1. Install [VS Code](https://code.visualstudio.com/) and the **Python**
   extension (Microsoft) from the Extensions panel (`Cmd/Ctrl+Shift+X`).
2. Open the `DiscEvolution` folder in VS Code (`File > Open Folder`).
3. Point VS Code at your virtual environment: `Cmd/Ctrl+Shift+P` →
   **"Python: Select Interpreter"** → pick the one inside `.venv`. Without
   this step VS Code will use your system Python and won't find the
   packages you just installed.

### Using an AI coding assistant

Either of these is genuinely useful for a codebase this size — use them to
get oriented, not just to write code.

- **GitHub Copilot**: install the *GitHub Copilot* and *GitHub Copilot Chat*
  extensions, sign in with your GitHub account. Inline gray-text suggestions
  appear as you type (accept with `Tab`); the chat panel (`Ctrl+Alt+I` /
  `Cmd+Ctrl+I`) is better for questions like *"what does `DustGrowthTwoPop`
  do?"* or *"why is this throwing a KeyError?"*.
- **Claude Code**: install the *Claude Code* extension from the VS Code
  marketplace, sign in, and open its chat panel from the sidebar. It can
  read the whole repo, not just the open file, so it's good for questions
  that span files, e.g. *"trace how `psi_DW` flows from the config file to
  the equation of state"*. Type `/` in its chat box to see available
  commands. Treat it like a knowledgeable but occasionally-wrong labmate:
  useful for explanations and first-draft code, but check anything it tells
  you about the physics against the actual DiscEvolution paper/code.

Either tool is well suited to: explaining an unfamiliar function, tracing
where a variable's value comes from, drafting a docstring, or figuring out
why a traceback happened. Be more careful trusting either one to explain
*why* a piece of physics is implemented a particular way — verify against
the code and, ideally, the papers it's based on.

### The Python debugger

Print statements get you only so far. The debugger lets you pause execution
and inspect every variable at that point — much faster for understanding
this codebase's units and data shapes than reading in isolation.

1. Open `run_model_student.py`. Click in the gutter (left of the line
   numbers) on a line inside `run_model()` or `_integrate()` to set a
   **breakpoint** (a red dot appears).
2. Open the **Run and Debug** panel (`Cmd/Ctrl+Shift+D`) → **"create a
   launch.json file"** → **"Python File"**. This creates
   `.vscode/launch.json` (already gitignored, so it's just for you).
3. Edit it to pass a small config and a short run, e.g.:
   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Debug run_model_student",
         "type": "debugpy",
         "request": "launch",
         "program": "${file}",
         "args": ["--config", "config/DiscConfig_default.json"],
         "console": "integratedTerminal",
         "cwd": "${workspaceFolder}/example/StartHere"
       }
     ]
   }
   ```
4. Press `F5` (or the green ▶ in the Run and Debug panel). Execution stops
   at your breakpoint. Hover over any variable to see its value, use the
   **Debug Console** at the bottom to evaluate expressions (e.g. type
   `disc.Sigma.max()`), and use the step buttons (step over / into / out)
   to advance line by line.

For a quick single-file smoke test without setting up `launch.json`, edit
`config/DiscConfig_default.json` to a tiny grid (`"nr": 30`) and short run
(`"t_final": 0.01`), then just use VS Code's **"Python: Debug Python File"**
default configuration.

---

## 4. Daily git workflow

Commit **small and often** — after each piece of working progress, not just
at the end of the day. Small commits are easier to review, easier to revert
if something breaks, and make it much easier for someone else (or you, in
three months) to understand what changed and why.

```bash
git checkout -b my-feature-name        # start a new branch for each task
# ... edit files ...
git status                             # see what changed
git add path/to/file.py                # stage specific files (avoid `git add .`
                                        #  blindly -- check you're not adding
                                        #  output/ or logs/, though .gitignore
                                        #  already excludes those)
git commit -m "Short, specific description of what changed and why"
git push -u origin my-feature-name     # first push of this branch
```

Periodically pull in updates from the shared repo:
```bash
git fetch upstream
git merge upstream/master               
```

When a piece of work is ready for feedback, open a **pull request** on
GitHub from your branch — either against your own fork (if just checkpointing
for yourself) or against `mlbalogh/DiscEvolution` (if it's ready to share).
Small, focused PRs get reviewed faster than one enormous one.

A good rule of thumb: if you can't describe what a commit does in one short
sentence, it's probably doing too many unrelated things — split it up.

---

## 5. Run a single model

Everything in `example/StartHere/` assumes your virtual environment is
active (`source .venv/bin/activate` from the repo root) and that you `cd`
into `example/StartHere` first.

```bash
cd example/StartHere
python run_model_student.py --config config/DiscConfig_default.json \
    --psi_DW 0.01 --Mdot 1e-8 --M 0.1 --Rd 50
```

- `--config` points at a JSON file describing everything about the run
  (grid, star, chemistry, planets, ...) — see `DiscConfig_default.json` for
  a fully-commented-by-example version of most fields.
- `--psi_DW`, `--Mdot`, `--M`, `--Rd`, `--output_dir` override individual
  values from that file, which is how `run_popsynth_student.sh` sweeps a
  grid without needing a separate config file per run.
- Output goes to `--output_dir`, or `$DISCEVOLUTION_OUTPUT` if set, or
  `config["simulation"]["output_dir"]`, or `./output` — in that priority
  order — as one HDF5 (`.h5`) file per run.

**Before a long run**, always test with a tiny, fast version first: copy the
config, shrink `grid.nr` to ~30-50 and `simulation.t_final` to something
that finishes in seconds, and confirm it runs cleanly end to end. Real
production grids (`nr: 1000`, `t_final` of a few Myr) can take hours.

If you need to modify the .json file, create a new file with a helpful name.
This way you can always reconstruct a simulation - if you just overwrite entries
that information is lost.

If you re-run the exact same parameters, the script notices the output file
already exists and is complete, and skips the work instead of redoing it
(see [Section 7](#7-batch-runs-parameter-sweeps)).

---

## 6. Code architecture & units

### Pipeline

`run_model_student.py` runs one simulation, top to bottom:

```
config (JSON)
   │
   ├─ Grid + SimpleStar                              (grid, star)
   ├─ disc_setup.setup_disc()                         solve for the initial
   │                                                   disc structure (Σ, T, α)
   ├─ build_transport() + build_dust_growth_disc()     gas/dust/diffusion
   │                                                   operators, wrap disc
   │                                                   in DustGrowthTwoPop
   ├─ build_chemistry()                                seed ice/gas abundances
   ├─ build_planetesimals()                            (optional)
   ├─ build_planets()                                  (optional)
   ├─ create_output_file()                             open the HDF5 file
   └─ _integrate()                                      the actual timestep
                                                          loop, streaming
                                                          snapshots to disk
```

`disc_setup.py` holds *only* the disc-initialization math that solves for
alpha and Sigma given disk mass, size and Mdot.  Wee its module
docstring for how to add another method to do something different. 

### Units — read this before you trust any number

DiscEvolution works in units where **G = 1**, most lengths are in **AU**, and
most masses are in **Msun** (see `DiscEvolution/constants.py`). Those three
choices fix the time unit too, via Kepler's third law — the constant `yr`
(= 2π) converts between it and real years:
```python
t_code  = t_years * yr      # years -> code time
t_years = t_code  / yr      # code time -> years
```
That's why you'll see `* yr` and `/ yr` scattered everywhere `t` is printed
or compared to a config value in years.

Quick reference for the quantities you'll meet most:

| Quantity                     | Units                         |
|-------------------------------|-------------------------------|
| `R`, `Rd`, `grid.Rc`           | AU                             |
| `Sigma` (any of them)          | g / cm²                        |
| Disc mass (`disc.Mtot()`, `M` in config) | grams internally; config values are in **Msun** |
| `Mdot`                          | Msun / yr                      |
| `T`                              | K                               |
| `alpha`, `alpha_SS`             | dimensionless                  |
| `t` (in this script)             | code-time units — divide by `yr` for years |
| Planet core/envelope mass (`M_core`, `M_env`) | **Mearth**, not Msun! |

**⚠️ This is not fully consistent across the codebase — some genuine traps:**

- `disc_params['M']` (disc mass) in the config is in **Msun**, but
  `disc.Mtot()` returns **grams**; the code divides by the `Msun` constant
  wherever it needs to compare them. If a number involving mass looks off
  by ~30 orders of magnitude, this is almost always why.
- **Disc** mass is in Msun; **planet** mass is in **Mearth**. There's no
  automatic conversion — mixing these up silently gives numbers that are
  wrong by a factor of ~333,000 (`Msun/Mearth`) without erroring.
  `DiscEvolution/constants.py` has `Mjup = 317.8 * Mearth` if you need to
  convert.
- The `-2*np.pi*R*Sigma*v*(AU*AU)*(yr/Msun)` pattern you'll see for
  converting a local accretion-rate calculation to Msun/yr mixes a
  geometric `2*np.pi` (the standard `Ṁ = -2πRΣv` flux formula) with a
  *separate* `yr`/`Msun` unit-conversion factor in the same expression.
  They look similar but mean different things — don't "simplify" one
  into the other.
- `alpha` (total) vs `alpha_SS` (viscous-only) differ by a factor of
  `(1 + psi)` whenever a disc wind is on (`psi_DW > 0`) — the EOS wants
  `alpha_SS`, config files specify `alpha` (total).

When in doubt: use the debugger (Section 3) to check a variable's actual
magnitude against this table rather than guessing.

### Output format

Every run produces one HDF5 file with a fixed set of dataset names
(`Sigma_G`, `T`, `Mcs`/`Mes`/`Rp` per planet, chemistry abundances, etc.) —
see the comment block above `create_output_file()` in `run_model_student.py`
for the full list. **Don't rename these** if you modify the script; the
loader in `notebooks/` (e.g. `HJpaper.ipynb`), key off these exact names.

---

## 7. Batch runs (parameter sweeps)

`run_popsynth_student.sh` runs a grid of models (over `psi_DW`, `Mdot`, `M`,
`Rd`) in parallel:
```bash
cd example/StartHere
./run_popsynth_student.sh
```
Edit the four `..._VALUES` lines near the top of the script to change what
gets swept, and `CONFIG_FILE` if you want a different base config.

Python side checks for itself whether that exact output already exists and
is complete — if so it prints "Skipping" and exits immediately instead of
re-simulating. That means re-running `run_popsynth_student.sh` after an interrupted 
sweep (crashed laptop, killed job, etc.) is always safe — completed runs are skipped,
incomplete/missing ones are (re)done.

---

## 8. Running fully in the background

For a sweep that takes hours, you don't want it tied to your terminal
session — if you close your laptop or lose your ssh connection, a normal
background job (`command &`) gets killed. Two extra pieces fix that:

```bash
nohup setsid ./run_popsynth_student.sh > master.log 2>&1 &
```

- **`nohup`** — makes the process ignore the "hangup" signal your terminal
  sends to its children when it closes.
- **`setsid`** — starts the process in a brand-new session, fully detached
  from your terminal's process group, so it isn't affected by that terminal
  at all (not just hangups).
- **`> master.log 2>&1`** — since nothing is printing to your screen
  anymore, redirect both normal output (`>`) and errors (`2>&1`) to a log
  file you can check later.
- **`&`** — runs it in the background so you get your prompt back
  immediately.

Afterwards:
```bash
tail -f master.log          # watch progress live (Ctrl+C just stops watching,
                             #  doesn't stop the job)
ps aux | grep run_popsynth  # confirm it's still running
kill <PID>                  # if you need to stop it (PID from `ps`, or `jobs -l`
                             #  right after launching, before you disconnect)
```
Per-run logs also land in `example/StartHere/logs/<tag>.out` /
`.err` — check those first if one particular parameter combination fails.

**Mac / Linux:** works exactly as above, in Terminal or over ssh.

**Windows:** `nohup`/`setsid` don't exist natively, and this repo's batch
scripts are bash + (optionally) GNU `parallel`. The straightforward path is
**WSL2** (Windows Subsystem for Linux):
1. In PowerShell (as Administrator): `wsl --install`, then restart.
2. Open the "Ubuntu" app it installs — that's a real Linux terminal.
3. Redo Sections 1–2 of this guide *inside* WSL (clone the repo into your
   WSL home directory, create the venv there) — from that point on, every
   command in this guide, including `nohup setsid ...`, works completely
   unmodified.

If you'd rather stay in native Windows for a single foreground run (not a
long detached batch job), `python run_model_student.py ...` in PowerShell
works fine on its own — it's only the *bash batch launcher* and the
*detached-background trick* that need WSL (or Git Bash, for the launcher
only — `nohup`/`setsid` still won't be available there).

---

## 9. Where to go next

- **Other disc setups** (fixed-Rd, no winds, etc.): `run_model_discchem_stream.py`
  and `disc_setup.py`'s module docstring.
- **Analysing output**: see the notebook `HJpaper.ipynb`.
- **Questions**: don't sit stuck on a plumbing/environment issue for more
  than ~15 minutes — ask. Physics questions about *why* the code does
  something a particular way are worth spending more time on yourself
  first, ideally with the paper open next to the code.
