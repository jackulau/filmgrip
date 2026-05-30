# film-grip — editor capability matrix

Generated from `filmgrip.adapters.registry`. film-grip surfaces these so it never promises an editor automation it cannot deliver.

| Editor | Role | Live selection | Write-back | Needs app | Mechanism |
|---|---|---|---|---|---|
| DaVinci Resolve (Studio) | flagship-native | no | yes | yes | native Python scripting API (fusionscript) |
| Final Cut Pro | interchange | no | yes | no | FCPXML round-trip (File ▸ Export/Import XML) |
| Premiere Pro | interchange | no | yes | no | FCP7 XML / AAF round-trip (UXP panel = future path) |
| Avid Media Composer | best-effort | no | yes | no | AAF interchange (relink/conform) |
| Kdenlive | interchange | no | yes | no | native MLT XML (.kdenlive) parse/rewrite |
| Shotcut | interchange | no | yes | no | native MLT XML (.mlt) parse/rewrite |
| CapCut (International) | best-effort | no | yes | no | offline draft_content.json rewrite (microsecond timeranges) |
| Wondershare Filmora | read-only | no | NO | no | offline .wfp (ZIP of JSON/XML) READ-ONLY parse |
