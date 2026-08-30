# Releasing BabelBrain

BabelBrain ships as **two small apps** so users can pick and swap between
BabelBrain versions without reinstalling:

* **`BabelBrain.app`** — the main app; opens BabelBrain directly by running the
  currently-selected version.
* **`BabelBrain-Version-Selector.app`** — the picker: choose, download, and
  switch versions. Selecting a version makes it the one `BabelBrain.app` runs.

Neither app embeds a BabelBrain version; the actual frozen versions live in a
per-user / shared **versions store**, into which the installer seeds a default.
This document covers how a release is built and how a version becomes selectable.

If you just want the app architecture, see `CLAUDE.md`. The launcher design
lives in the module docstrings under `BabelBrain/Hub/`.

---

## What a build produces

The GitHub Actions workflow `.github/workflows/build-release.yml`, per platform,
produces:

1. **The installer** — packages BOTH apps and seeds a default version:
   - macOS: a signed/notarized `.dmg` containing a `.pkg` that installs
     `BabelBrain.app` + `BabelBrain-Version-Selector.app` to `/Applications` and
     seeds the version into `/Users/Shared/BabelBrain/versions/<build_id>/`.
   - Windows: a per-user Inno Setup `.exe` (no admin) that installs both apps and
     seeds the version into `%LOCALAPPDATA%\BabelBrain\versions\<build_id>\`.
   The seeded version makes a fresh install work offline.

2. **A relocatable version bundle** — `‹artifact›-version.zip` plus
   `‹artifact›-version.zip.sha256`. This is what the Version Selector downloads
   to add or swap versions. It is a standalone, (on macOS) notarized BabelBrain
   bundle with an embedded `build_info.json`.

`‹artifact›` is one of `BabelBrain-macOS-arm64`, `BabelBrain-macOS-x64`,
`BabelBrain-Windows-x64`.

---

## How a build identifies itself in the UI

A non-stable build says so in the main window title, so a tester never has to
guess which build they are looking at:

| Built from | `channel` | Window title |
| --- | --- | --- |
| `v*` tag (release) | `stable` | `BabelBrain V0.8.8 - …` |
| `test-*` tag | `prerelease` | `BabelBrain V0.8.8 [test-0.8.8b2-c0cf1e6 · 2026-08-30] - …` |
| manual dispatch | `dev` | `BabelBrain V0.8.8 [dev-c0cf1e6 · 2026-08-30] - …` |
| `create_unsigned_dmg.sh` | `dev` | `BabelBrain V0.8.8 [dev-c0cf1e6 · 2026-08-30] - …` |
| **running from source** | — | `BabelBrain V0.8.8 - …` (never labelled) |

`BabelBrain.py:GetBuildLabel()` reads the `build_info.json` inside the *running*
bundle — the app's own identity, not the Hub's idea of what it launched, so the
label is right even when a version is started directly out of the store. It is
gated on `sys.frozen`: a source checkout carries a git-tracked `build_info.json`
describing whatever was packaged last, which would mislabel the working tree.

The same label is appended to the telemetry `APP_VERSION` (`0.8.8 [dev-c0cf1e6 ·
2026-08-30]`), so dev builds are distinguishable from the release of the same
version in the logs. It is deliberately **not** added to the Brainsight
`# Created by: BabelBrain <version>` header, which stays a bare version number
that Brainsight can match against its approved list.

---

## Where versions are stored on the user's machine

Both apps read the same **versions store**. Each version lives in its own
`<build_id>` folder (e.g. `0.8.1+ca32b99/`) holding `BabelBrain.app` (macOS) or
`BabelBrain.exe` and its support files (Windows). The store has two roots, and
the Version Selector lets the user choose which to install into ("Just for me"
vs "For all users"):

| Scope | macOS | Windows | Linux (from source) |
| --- | --- | --- | --- |
| **Just for me** (per-user, no admin) | `~/Library/Application Support/BabelBrain/versions/` | `%LOCALAPPDATA%\BabelBrain\versions\` | `~/.local/share/BabelBrain/versions/` |
| **For all users** (shared) | `/Users/Shared/BabelBrain/versions/` | `%ProgramData%\BabelBrain\versions\` | `/opt/BabelBrain/versions/` |

Notes:

- **Elevation:** "For all users" needs admin on every platform in practice. On
  macOS `/Users/Shared` is itself world-writable, but the PKG seeds the store as
  **root**, so an existing shared store is root-owned and both installing and
  *removing* a version there prompt for administrator privileges (an
  authorization dialog via `osascript`); Windows (`%ProgramData%`) and a
  system-wide Windows install use a UAC-elevated helper, Linux (`/opt`) has no
  prompt. If elevation is denied the Selector shows the error and returns the
  user to the location choice — it never silently falls back to the per-user
  location, and a declined prompt is treated as a cancel, not a failure.
- **Both roots are always scanned**, so a version installed for all users and
  one installed just for the current user both appear in the picker, tagged by
  location.
- **The installer seeds the default version** here (not inside either app):
  macOS PKG → `/Users/Shared/BabelBrain/versions/<build_id>/`; Windows Inno →
  `%LOCALAPPDATA%\BabelBrain\versions\<build_id>\`. This is why a fresh install
  works offline.
- **The seeded version becomes the default.** The installer also writes a marker
  naming the build it just seeded — `/Users/Shared/BabelBrain/default_build.json`
  (macOS PKG postinstall, generated by `BabelBrain/Hub/make_pkg_scripts.sh`) or
  `%LOCALAPPDATA%\BabelBrain\default_build.json` (Inno `[Code]`). The first Hub
  start after the install adopts it as the current selection, once per user
  (`Hub/state.py:adopt_installer_default`), so a new build — a dev build in
  particular — actually runs instead of leaving the previous selection in place.
  A selection the user makes afterwards is never overwritten until the next
  install.
- **Small state** (the current selection, cached manifest) lives separately in
  `~/.config/BabelBrain/` (`hub.yaml`, `manifest_cache.json`) on all platforms —
  never in the versions store.
- The two launcher apps themselves install to `/Applications` (macOS) or the
  per-user `%LOCALAPPDATA%\Programs\BabelBrain` (Windows); only the heavy
  versions live in the store above.

These locations are defined in `BabelBrain/Hub/paths.py` — change them there if
needed (the installer seed paths in `build-release.yml` / `BabelBrain.iss` must
match).

---

## Cutting a release

Releases are driven by the git ref:

| Trigger | Channel stamped | GitHub Release |
| --- | --- | --- |
| Push tag `v*` (e.g. `v0.8.8`) | `stable` | Draft Release created with all assets |
| Push tag `test-*` | `prerelease` | **No** Release — artifacts only (for trials) |
| Manual "Run workflow" | `dev` | Only if `publish_release` is checked |

The channel and tag are written into each build's `build_info.json`
(`BabelBrain/Hub/gen_build_info.py`), which gives every build a stable identity
of `(version, git_commit)` — this is why a development `0.8.8` and a released
`0.8.8` are treated as distinct builds.

> **Always push a `test-*` tag first** to shake out the build/sign/notarize
> pipeline before pushing a real `v*` tag. `test-*` builds upload artifacts
> without publishing a Release.

The Release is always created as a **draft** so you can add notes before
publishing.

### macOS signing secrets

Signing/notarization is skipped (an unsigned DMG is still produced) unless these
repo secrets are set: `MACOS_CERT_P12_BASE64`, `MACOS_CERT_P12_PASSWORD`,
`MACOS_SIGN_IDENTITY`, `MACOS_INSTALLER_SIGN_IDENTITY`, `MACOS_NOTARY_APPLE_ID`,
`MACOS_NOTARY_TEAM_ID`, `MACOS_NOTARY_APP_PASSWORD`.

---

## Making a version selectable in the launcher (`releases.json`)

The Hub does **not** read the GitHub Releases API. It fetches a curated
manifest, `releases.json`, from the **default branch (`main`)**:

```
https://raw.githubusercontent.com/ProteusMRIgHIFU/BabelBrain/main/releases.json
```

Only versions listed there appear in the launcher. This keeps full control over
what users see — pre-releases published on GitHub stay hidden unless you add
them with `"channel": "prerelease"`.

### To activate a version

1. **Build it with the current pipeline** (push `v‹x›` or a `test-‹x›` tag) so a
   `‹artifact›-version.zip` + `.sha256` are attached to the release.
2. **Fill in the entry** in `releases.json` on `main`:
   - `url` → the download URL of that platform's `‹artifact›-version.zip`
   - `sha256` → the contents of the matching `‹artifact›-version.zip.sha256`
3. **Commit to `main`.** The Hub picks it up on next launch.

An entry with an **empty `url` stays hidden** in the launcher — so placeholder
entries are safe to keep until their assets exist.

### Generating an entry

Instead of hand-editing, use the helper and paste its output into the
`versions` array:

```bash
python BabelBrain/Hub/gen_manifest_entry.py \
  --version 0.8.8 --commit 0799425 --channel stable --recommended \
  --notes-url https://github.com/ProteusMRIgHIFU/BabelBrain/releases/tag/v0.8.8 \
  --asset macos-arm64=<url>:<sha256> \
  --asset macos-x64=<url>:<sha256> \
  --asset windows-x64=<url>:<sha256>
```

### Entry fields

| Field | Required | Meaning |
| --- | --- | --- |
| `version` | yes | Shown in the launcher, e.g. `"0.8.8"`. |
| `git_commit` | recommended | Short commit; with `version` forms the build id (`0.8.8+0799425`) used to dedupe and remember choices. |
| `channel` | no (`stable`) | `stable` or `prerelease`; pre-releases are hidden unless the user ticks "Show pre-releases". |
| `recommended` | no (`false`) | Tags the row "recommended". |
| `notes_url` | no | Link to release notes. |
| `assets` | yes | One entry per platform key (`macos-arm64`, `macos-x64`, `windows-x64`), each `{ "url", "sha256" }`. A version only appears on a platform it has an asset for. |

To **hide** a version, remove its entry (the GitHub Release can stay).

### Note on pre-existing (old) releases

Releases made before the Hub existed only carry the old installer assets
(DMG/exe), which the Hub cannot install — it needs the relocatable
`‹artifact›-version.zip`. To offer an old version through the Hub it must be
rebuilt and repackaged with the current pipeline; for much older versions this
may require their original Python/conda environment and may not be practical.
Such versions can remain as placeholder entries (empty `url`, hence hidden) for
documentation, or be dropped.

---

## Updating the Hub launcher itself

The Hub is versioned independently (`BabelBrain/Hub/__init__.py: HUB_VERSION`
and `BabelBrain/BabelBrainHub.spec: hub_version`). The `hub` block in
`releases.json` drives a non-blocking "a newer launcher is available" banner:

```json
"hub": { "latest": "1.1.0", "min_supported": "1.0.0", "url": "...", "sha256": "..." }
```

Because new BabelBrain versions only appear after the launcher is current, you
can use `min_supported` / `latest` to gate manifest-schema or feature changes.
v1 self-update is **notify-only** — users are pointed at the new installer;
there is no in-place auto-swap.
