# Dev vs Production (READ BEFORE RESTARTING THE SERVER)

Production Hark serves from **~/hermes-webui-prod** (pinned commit) under
launchd: `ai.tensor-systems.hark-webui` (KeepAlive — it resurrects itself).
Tailnet URL → Tailscale serve → 127.0.0.1:8787 = PRODUCTION.

## Production (port 8787)
- Restart: `launchctl kickstart -k gui/$(id -u)/ai.tensor-systems.hark-webui`
- Stop (for real): `launchctl bootout gui/$(id -u)/ai.tensor-systems.hark-webui`
- Logs: ~/hermes-webui-prod/logs/server.{log,err.log}
- Env/secrets: baked into the plist (launchd), sourced from ~/hermes-webui/.env + prod-env
- Deploy: commit in ~/hermes-webui, test, then in the PROD worktree:
  `cd ~/hermes-webui-prod && git merge --ff-only <tested-commit>` and kickstart.

## Dev (port 8788)
- `cd ~/hermes-webui && HERMES_WEBUI_PORT=8788 ./ctl.sh start --host 127.0.0.1 8788`
- NEVER bind dev to 8787 — that's production. NEVER `kill $(lsof -t -iTCP:8787)`.

## launchd PATH (Oct 1 lesson)
The LaunchAgent PATH is minimal (/usr/bin:/bin:/usr/sbin:/sbin) — it deliberately now includes
~/.hermes/tools/ffmpeg-9.0.1-darwin-arm64. Without ffmpeg, CAF→WAV conversion falls to afconvert,
which writes WAVE_FORMAT_EXTENSIBLE (tag 0xFFFE) for Float32 sources — ModelRelay's groq STT
rejects extensible WAV ('WAV must use PCM or IEEE float encoding') → every Hark utterance 400s.
If ffmpeg moves/updates, update PATH in the plist and `launchctl bootout` + `bootstrap`.
