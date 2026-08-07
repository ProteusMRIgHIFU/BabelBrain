# Releasing BabelBrain

BabelBrain ships as a small **Hub launcher** that lets users pick and swap
between BabelBrain versions without reinstalling. This document covers how a
release is built and how a version becomes selectable in the launcher.

If you just want the app architecture, see `CLAUDE.md`. The Hub's own design
lives in the module docstrings under `BabelBrain/Hub/`.

---

## What a build produces

The GitHub Actions workflow `.github/workflows/build-release.yml`, per platform,
produces:

1. **The installer** — the Hub launcher packaged for users:
   - macOS: a signed/notarized `.dmg` containing a `.pkg` (installs to
     `/Applications`).
   - Windows: a per-user Inno Setup `.exe` (no admin required).
   The installed app **is the Hub**; it carries a read-only "bundled" copy of
   that build's BabelBrain so a fresh install works offline.

2. **A relocatable version bundle** — `‹artifact›-version.zip` plus
   `‹artifact›-version.zip.sha256`. This is what the Hub downloads to add or
   swap versions. It is a standalone, (on macOS) notarized BabelBrain bundle
   with an embedded `build_info.json`.

`‹artifact›` is one of `BabelBrain-macOS-arm64`, `BabelBrain-macOS-x64`,
`BabelBrain-Windows-x64`.

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
