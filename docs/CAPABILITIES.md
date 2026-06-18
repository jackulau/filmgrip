# film-grip — editor capability matrix

Generated from `filmgrip.adapters.registry`. film-grip surfaces these so it never promises an editor automation it cannot deliver.

**Audio**: live = import+place on an audio track via scripting; interchange/offline = via file round-trip; read-only = parsed, not written. **Per-clip volume/gain/fades are NOT scriptable in Resolve** (Fairlight-only) — film-grip places audio, levels stay manual.

**Color**: the scriptable grading surface is `set_cdl` (ASC CDL primary grade — the portable pivot, rides through OTIO metadata + `.cc`/`.ccc`/`.cdl` sidecars + EDL + FCPXML), `apply_lut` (a baked look), `color_group`/`color_version` (Resolve-live organization), and `apply_grade` (copy a hero grade / apply a `.drx` PowerGrade). Perception (`film-grip scopes`) synthesizes parade/waveform/vectorscope/white-balance since Resolve exposes no scope API. **Primary wheels by value, curves, qualifiers, Power Windows, Magic Mask and the AI color tools are GUI-only in every NLE — not scriptable** — so film-grip surfaces them as advisory steps, never as applied edits.

| Editor | Role | Write-back | Audio | Organize | In-app panel | Selection | Mechanism |
|---|---|---|---|---|---|---|---|
| DaVinci Resolve (Studio) | flagship-native | yes | live | live | native | reconstructed | native Python scripting API (fusionscript) |
| Final Cut Pro | interchange | yes | interchange | interchange-warn | none | precise | FCPXML round-trip (File ▸ Export/Import XML) |
| Premiere Pro | interchange | yes | interchange | interchange-warn | uxp-future | precise | FCP7 XML / AAF round-trip (UXP panel = future path) |
| Avid Media Composer | best-effort | yes | interchange | interchange-warn | none | precise | AAF interchange (relink/conform) |
| Kdenlive | interchange | yes | interchange | interchange-warn | none | precise | native MLT XML (.kdenlive) parse/rewrite |
| Shotcut | interchange | yes | interchange | interchange-warn | none | precise | native MLT XML (.mlt) parse/rewrite |
| CapCut (International) | best-effort | yes | offline | none | none | precise | offline draft_content.json rewrite (microsecond timeranges) |
| Wondershare Filmora | read-only | NO | read-only | none | read-only | readonly | offline .wfp (ZIP of JSON/XML) READ-ONLY parse |

## Which ops land where

| Op | Resolve (live) | Resolve (rebuild) | Interchange file |
|---|---|---|---|
| trim | rebuild | yes | yes |
| move | rebuild | yes | yes |
| split | rebuild | yes | yes |
| cut_range | rebuild | yes | yes |
| insert | rebuild | yes | yes |
| ripple | rebuild | yes | yes |
| delete | yes | yes | yes |
| retime | rebuild | yes | yes |
| add_marker | yes | yes | yes |
| set_property | yes | yes | yes (metadata) |
| set_enabled | yes | yes | yes |
| add_transition | no | no | no (do in editor) |
| import_audio | yes | n/a | no |
| add_track | yes | n/a | no |
| rename_track | yes | n/a | no |
| create_bin | yes | n/a | no |
| move_to_bin | yes | n/a | no |
| set_cdl | yes | yes | yes |
| apply_lut | yes | yes | yes |
| color_group | yes | no | no |
| color_version | yes | no | no |
| apply_grade | yes | no | no |
