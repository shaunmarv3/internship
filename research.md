# Research Notes — Maritime Security Intelligence

Running log for the paper write-up. Newest sections appended at the bottom.

---

## Sensors, AIS sync & data provenance (2026-06-23)

### Optical vs SAR — why we use SAR, not Google Earth
There are two fundamentally different satellite imaging types. We deliberately use SAR.

| | Optical (Google Earth) | SAR (our datasets) |
|---|---|---|
| Instrument | Camera — reflected sunlight (passive) | Radar — microwave echo (active) |
| Appearance | Colorful RGB | Grayscale (radar has no color) |
| Satellites | Sentinel-2, Landsat, Maxar, Airbus | Sentinel-1, TerraSAR-X |
| Night / cloud | Blind | Sees through clouds, works day & night |
| Freshness | 1–3 years old, ~monthly piecemeal, NOT real-time | Days, near-real-time |

- An "optical sensor" *is* a camera. Google Earth carries **no radar**; imagery averages **1–3 years old** and updates piecemeal (~monthly) — explicitly **not** real-time. Sources: mygpstools Google Earth update guide; geowgs84 "how old are Google satellite images".
- **Paper argument:** Google Earth is unusable for live maritime surveillance — stale + cloud-blocked. SAR is grayscale but fresh, all-weather, day/night → the standard for dark-vessel detection.
- **Important distinction — Google Earth (app) vs Google Earth *Engine* (platform):** the consumer Google Earth basemap is optical-only. Google Earth *Engine* is a separate data+compute platform whose catalog **DOES include Sentinel-1 SAR** (`COPERNICUS/S1_GRD`): C-band dual-pol GRD, updated daily, 6-day revisit, 10/25/40 m, delivered **in dB** (thermal-noise removal → calibration → terrain correction). Source: Earth Engine Data Catalog (developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD).
- GEE `S1_GRD` is Sigma0 in **dB** — same format as the Zenodo oil data → `sar_preprocess.py` (dB→linear→Lee→normalize) runs identically on live GEE pulls and Zenodo. This IS the "live/GEE" inference path. Export with `geemap.ee_export_image(..., scale=10, region=roi)` keeps CRS+affine → enables pixel→lat/lon. Caveat: analysis-ready within hours–days, not live-this-second; revisit 6 days.
- Confirm visually in GEE by overlaying Sentinel-2 (`COPERNICUS/S2_SR_HARMONIZED`, bands B4/B3/B2 = color) vs Sentinel-1 (`COPERNICUS/S1_GRD`, VV = grayscale) on the same ROI.

### Dataset provenance (verified)
- **HRSID (ships):** 99 Sentinel-1B + 36 TerraSAR-X + 1 TanDEM-X scenes, cropped to 5,604 tiles @ 800×800; resolutions 0.5/1/3 m; polarizations HH/HV/VV; COCO-format JSON annotations; 16,951 ship instances. Note: NOT pure Sentinel-1 — mostly Sentinel-1 augmented with higher-res TerraSAR-X. Source: HRSID paper (Wei et al.).
- **Oil (Zenodo, records 8346860 / 8253899 / 13761290):** Sentinel-1 C-band, Sigma0 in **dB**, 2 polarizations VV/VH, 2048×2048, georeferenced. Confirmed by inspector (min ≈ −30 → dB) and Part III record text.

### GEE Sentinel-1 preprocessing (confirmed from EE "Sentinel-1 Algorithms" doc)
`COPERNICUS/S1_GRD` = Level-1 GRD → backscatter coefficient σ° in **dB** (`10·log10(σ°)`). EE applies, via the Sentinel-1 Toolbox:
1. apply orbit file  2. GRD border-noise removal  3. thermal-noise removal  4. radiometric calibration  5. terrain correction (orthorectification, SRTM 30 m / ASTER DEM > ±60° lat).
EE does **NOT** apply speckle filtering, and leaves values in **dB**. → our `sar_preprocess.py` (dB→linear→Lee→normalize) is the correct complement and makes a live GEE pull match the Zenodo-dB training data exactly (no train/serve skew). Filter to a homogeneous subset first: `transmitterReceiverPolarisation` ['VV','VH'], `instrumentMode`='IW', `orbitProperties_pass` ASC/DESC. Verified live pull worked: scene acquired 2024-01-01 01:03:31 UTC, Sigma0 dB VV/VH. Note: SLC (complex/phase) not ingestible in EE; GRD only.

### SAR preprocessing (sar_preprocess.py — same for train & inference)
1. raw Sentinel-1 Sigma0 **dB → linear**: `10^(dB/10)`
2. **Lee speckle filter** (SAR speckle noise)
3. **percentile-clip (1–99) → normalize [0,1]**
4. stack VV/VH → 3 channels, ImageNet-normalize for the encoder

HRSID JPEGs arrive already 8-bit grayscale (Sentinel-1 intensity → log → 0–255); YOLO just scales 0–255 → 0–1.

### Why oil scenes look like "speckle static" vs ship scenes (visual comparison)
Observed: oil VV tiles render as salt-and-pepper static, while HRSID/GEE tiles look like recognizable scenes. Reason (NOT a data fault):
- Oil scenes are **pure open ocean** — uniform very-low backscatter (stats: min ≈ −54 dB). No land/structures for contrast. Percentile-stretch on a near-uniform field amplifies SAR **speckle** → "static" look.
- HRSID/GEE comparison tiles were **coastal/urban** — bright buildings/docks dominate → speckle hidden, looks "real".
- **Inverse target signatures:** oil = smooth **DARK** patch (oil dampens waves → less backscatter); ship = bright **POINT** (metal → strong return). Oil detection vs ship detection are inverse anomaly problems on the same SAR. Good paper framing.
- Fix for display/model: apply **dB→linear + Lee filter** (sar_preprocess) → speckle smooths, slick becomes visible. Confirms the Lee step's value. Oil tile stats: 00000 min −54.1/max 5.5; 00002 min −50.3/max −14.6 (oil 0.39%); 00003 (oil 0.59%) — all dB, 2-band 2048², masks {0,1}. Note rasterio warning "no geotransform" → Zenodo tiles are NOT georeferenced (geo-ref only stated for Part III); fine for training, but live geo-coords come from GEE scenes which DO carry CRS.
- For apples-to-apples demo: pull GEE S1 VV over **open sea with ships** (bright dots on dark water) ≈ oil scene (dark slicks on dark water). Zoom = smaller `buffer` + larger `dimensions` in `getThumbURL`.

### CRITICAL FINDING: oil signal is in BAND 2 (VH), not BAND 1 (VV)
Measured oil-vs-water mean backscatter on real Part I tiles (00000/00002/00003/00004):
- **Band 1 (nominally VV): ~+1 dB** (00000 even −1.8 dB *brighter*) → essentially NO oil signal.
- **Band 2 (nominally VH): +6.0 / +8.2 / +9.1 / +9.4 dB DARKER** → strong, consistent oil signal.
So earlier "pure static" displays were band 1 (the uninformative channel). The slick is clearly separable in band 2 (~9 dB), and appears as a dark region matching the GT mask after a 15px multilook. Oil is MORE learnable than the band-1 panels suggested.
- **Band-order caveat:** dataset doc states order (VV, VH), but band-1 (~−36 dB) is *darker* than band-2 (~−21 dB), which is backwards from typical ocean (VV>VH). So the file's nominal VV/VH labels may be swapped, OR these are low-VV scenes — irrelevant for ML: the informative channel is **band 2** whatever its label. Physically consistent with oil damping co-pol Bragg scattering (~9 dB drop).
- **Training implication:** loaders.py stacks both bands → model already receives band 2 → signal is available. **APPLIED (2026-06-24):** changed `_load_image` 2-band stacking so the 3rd encoder channel duplicates **band 2 (VH, strong)** instead of band 1 (weak). Channels are now `[VV, VH, VH]` — keeps VV for look-alike/sea-state context while giving the ~9 dB oil signal 2 of 3 channels (was `[VV, VH, VV]`). One-line edit, no train/serve skew (inference uses same loader). Model could in principle learn the weighting itself, but this removes the handicap of feeding the weak band twice.
- Revises the earlier pessimism: oil signal is solidly present (9 dB). Still harder than ships (look-alikes: low-wind patches/algae mimic the same low-backscatter; dataset has a dedicated look-alike class). Expect modest but real oil IoU (~50-65% per literature).

### Why the band-2 fix is oil-only (no symmetric ship change) (2026-06-24)
The `[VV,VH,VH]` loader fix applies to M3 oil only — **not** M1 ships — for two reasons:
1. **No band to choose.** HRSID tiles are single-channel 8-bit grayscale JPEGs (mixed HH/HV/VV baked into one intensity image), replicated to 3 identical channels by YOLO. There is no second band to swap; the oil fix ("don't triplicate the weak band") presupposes 2 separate bands, which only the Zenodo GeoTIFFs have.
2. **Ships are polarization-robust** (bright point targets in every pol), so band choice is immaterial for detection — unlike oil, which is polarization-sensitive (signal only in VH).
- The only VH decision for ships is at **inference**: feed the detector the VH band of a live GEE dual-pol pull (cleanest — dark sea, ships pop), replicate to 3, run YOLO. Runtime serving choice for M6, not a training-code edit.
- **Paper point:** *oil is polarization-sensitive, ships are polarization-robust* — this asymmetry is itself a finding, and explains why the two pipelines treat polarization differently.

### VV vs VH polarization (why dual-pol matters)
Sentinel-1 transmits Vertical; 2nd letter = receive pol. VV = co-pol (same orientation back), VH = cross-pol (rotated 90°). Cross-pol return requires the target to DEPOLARIZE the wave (multiple bounces / complex structure).
- **Sea:** bright+grainy in VV (Bragg scattering off wind ripples preserves pol → clutter); near-black in VH (flat water barely depolarizes).
- **Ship:** bright in both, but in VH it stands on a black background → far higher ship-to-clutter contrast. Confirmed visually on GEE open-sea scene: VV grainy sea + ships; VH black sea + ships popping cleanly.
- **Use both:** VH = cleanest detection (ships; oil slick at ~9 dB); VV = sea-state/wind + surface texture context (helps separate real oil from low-wind look-alikes). Dual-pol input > single pol. Caveat: VH is weak (near noise floor) → ESA thermal-noise removal matters; calm seas make VH noisier.

### Ship detection — confusers (SAR sees radar reflectivity, NOT light)
SAR is radar: light-emitting objects (lighthouses, deck lights, flares) do NOT appear — only radar reflectors do. Real ship-confusers: offshore platforms/wind turbines/buoys (strong metal returns), small islands/rocks/coast (inshore FPs), very strong point targets + SAR sidelobe "star/cross" artifacts, sea ice/icebergs, RFI streaks. Discrimination: CNN learns ship shape/size/context (not raw brightness); persistent infrastructure masked via DB/temporal differencing; AIS cross-check; land mask. **Limitation:** HRSID is single-class "ship" → detector cannot natively separate ship vs platform; that's a downstream M6 (AIS + infrastructure mask) step.

### Pixel → geo-coordinate (M1 detection output)
YOLO returns a **pixel** box `(x,y,w,h)`. The full Sentinel-1 GRD GeoTIFF is georeferenced (CRS + affine), so:
`box centre (col,row) → src.xy() → map coords (often UTM) → warp to EPSG:4326 → lon/lat`.
Caveat: HRSID 800×800 tiles are JPEG crops with **no** geo-reference — they're for training detection only; the lat/lon step works on the full georeferenced scene.

### AIS cross-verification & latency
- **AIS** = ships broadcast MMSI/position/speed/heading over VHF. SAR detection with **no matching AIS** in space+time = **dark vessel**.
- **GFW free near-real-time pipeline: ~72-hour (3-day) processing delay**; ingests 110M+ AIS msgs/day. Source: GFW Data Availability, GFW APIs docs.
- Operational/commercial AIS (Spire, terrestrial) = seconds–minutes; satellite AIS = ~15 min–hours.
- Free sources for the project: GFW (already used in M2), NOAA Marine Cadastre, Danish Maritime Authority (all historical).
- **Sync logic:** SAR scene has exact UTC timestamp → interpolate AIS tracks to that instant → project onto scene → match within ~100–500 m gate → unmatched detections = dark ships. Because GFW is 72h late, match against a historical Sentinel-1 date.
- **Thesis reinforcement:** AIS is delayed *and* can be switched off → cannot rely on AIS alone → SAR sees the ship regardless. The fusion is the contribution.

### "Won't the detector flag everything as a ship?" (demo concern)
- The dense green-box scenes were the **busiest harbor tiles** (display sorted by ship count, descending) — worst case, not typical. Most tiles have ~1–15 ships.
- Boxes are labels/predictions, not real-world fixtures; at inference YOLO *produces* the boxes.
- Model trains on negatives too (empty sea, unboxed cities) → "bright ≠ ship". Ships = bright compact hard targets vs dark water.
- Genuine weakness: **inshore false positives** (docks, cranes, small islands). HRSID's inshore/offshore split lets us *report* this; production uses a coastline/land mask + confidence threshold + NMS.

### M3 training data protocol — all 3 parts (2026-06-24)
Decision: the reportable M3 model trains on **Part I + Part II** and is tested on **Part III** (held out). Roles:
- **Part I** (1,200 oil + masks) = positives.
- **Part II** (lookalike + no-oil + masks) = hard negatives — without them the model over-predicts oil (flags every dark patch); the dataset has a dedicated look-alike class for exactly this. Combined I+II ≈ 2,570 samples (matches train_segmentation.py docstring).
- **Part III** (150 oil / 150 lookalike / 150 no-oil + masks) = canonical TEST set → the paper benchmark number. NOT used for train/val (would be leakage + optimistic).
- **Why not Part I only:** a Part-I-only val split has no negatives → optimistic, non-reportable IoU. Part I alone was just a plumbing smoke check on real data.
- **Collision fix:** each part numbers from 0001 → naive merge collides filename stems and the loader (stem-based image↔mask pairing) would mis-pair. Prep prefixes stems per source: `p1_` (oil), `p2l_` (lookalike), `p2n_` (no-oil), `p3_` (test). Same prefix applied to a file's image and mask preserves pairing.
- **Part III split:** images+masks are mixed in one archive → separated by band count (2-band VV/VH = image, 1-band = mask), since the tiles aren't reliably foldered.
- Layout produced: `data/oil/{images,masks}` = I+II (train/val auto-split by loader); `data/oil_test/{images,masks}` = III. Eval loads `best_segformer.pt`, runs SegmentationMetrics on `data/oil_test`, reports OilIoU/mIoU (baseline_target 0.54).

### Data prep gotchas (Lightning, 2026-06-24) — all resolved
Building `data/oil` (I+II) and `data/oil_test` (III) surfaced several traps; all fixed and verified:
- **Extraction speed:** `py7zr.extractall` of the 40.7 GB Part I is very slow (pure-Python, single-thread). Use system `7z -mmt=on` (p7zip-full) — ~5-10x faster. The earlier "fast Colab" extract was a red herring (it only pulled 4 preview images, not 1,200).
- **HF rate-limit stall:** unauthenticated HF downloads throttle/stall after ~60 GB. Fix: `huggingface_hub.login(token)` → no-oil pull went 22.9 GB in 86 s @ 556 MB/s; Part III 9.9 GB @ 1.64 GB/s. Always authenticate for multi-archive pulls. `hf_hub_download` resumes partial `.incomplete` files.
- **Part III filename collisions (silent data loss):** Part III mixes oil/lookalike/no-oil each numbered 0000–0149. A single `p3_` prefix collapsed 450 files → 150 (same-stem `shutil.move` overwrites). Fix: per-category prefix `p3o_`/`p3l_`/`p3n_` (category detected from path: "look"→l, "no"+"oil"→n, "oil"→o).
- **Part III mask stem suffix:** Part III masks are `<n>_segmentation.tif` while images are `<n>.tif` → loader pairs by exact stem → 0 matches. Fix: strip `_segmentation` when organizing. (Parts I/II masks are bare-numbered, so train paired fine.)
- **Always verify PAIRS, not counts:** matching image/mask *counts* ≠ matching *stems*. The `{stem}` set-intersection check caught both Part III bugs (showed MATCHED STEMS: 0 despite 450/450 counts). Part III images+masks separated by band count (2=image, 1=mask).

### Status — DATA VALIDATED, ready to train (2026-06-24)
- **TRAIN `data/oil`** (I+II): 2,570 images = 2,570 masks, **paired 2570/2570**, format **dB** (min ≈ −48…−55, p99 ≈ −26…−31), masks {0,1}, no NaNs, oil pixel mean 1.11%.
- **TEST `data/oil_test`** (III): 450 paired, dB. Inspector verdict showing oil 0.00% is a sampling artifact (8 samples all `p3l_` lookalikes = all-zero masks); `p3o_` oil images do carry oil.
- **Loader confirmed correct:** `_load_image → preprocess_sar_bands` does `dB→linear (db_to_linear) → Lee → normalize` — matches `sar_preprocess` inference (no skew). The inspector's "add dB→linear" warning is STALE hardcoded text (fix landed in commit d37913d); ignore it. (TODO low-pri: delete that stale message from inspect_oil_data.py.)
- Smoke test (imports+preprocess+oil) PASS on band-2 loader (commit 2f5bf49) before real training.

### M2 vessel-detection data — HRSID download (2026-06-24)
- **Dataset:** HRSID (High-Resolution SAR Images Dataset) for ship detection. Single class (0 = ship), 800×800 SAR JPEG chips in **COCO format** (`train2017.json` / `test2017.json`). These tiles are JPEG crops with **no** geo-reference (training only; lat/lon comes from the full georeferenced scene at inference — see AIS section above).
- **Source:** Google Drive file id `1NY3ovgc-woDlNoQdyqzRB3t9McOBH5Ms`, ~614 MB zip (not the ~1.5 GB upper estimate). Pulled with `gdown` (Drive's large-file virus-scan confirm is handled by `gdown.download(id=...)`).
- **Location:** extracted to `data/vessels/hrsid/` — kept the existing **plural** `data/vessels` dir to stay parallel with `data/oil` (user wrote "data/vessel"; resolved to the existing convention rather than creating a near-duplicate singular dir). Gitignored via `/data/`.
- **Reproducer:** root `download_vessel_data.py` (pip-installs nothing; needs `gdown`). `--force` re-downloads, `--keep-zip` retains the archive (default deletes it post-extract). Replaces the Colab `!gdown … && !unzip` snippet with a Windows-friendly pure-Python `zipfile` extract (no `unzip` dependency).
- **Viewing all images:** `data/vessels/view_hrsid.py` — default mode saves `hrsid_overview.png` (the N busiest harbor scenes with lime ship boxes, reproducing the original snippet); `--gallery` builds `hrsid_gallery.html` linking **every** image with per-image ship counts for scroll-through browsing.
