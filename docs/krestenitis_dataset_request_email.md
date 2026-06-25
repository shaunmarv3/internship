# Krestenitis Oil Spill Dataset — Access Request Email

Send this from your **supervisor's institutional email**.
- **To:** mikrestenitis@iti.gr
- **CC:** kioannid@iti.gr
- **From:** [SUPERVISOR's institutional email — mandatory]

---

**Subject:** Dataset Access Request — Oil Spill Detection Dataset (M4D / ITI)

Dear Dr. Krestenitis and Dr. Ioannidis,

I am [SUPERVISOR NAME], [TITLE] at [INSTITUTION], and I am writing to request access to the Oil Spill Detection Dataset published by your group at the Multimedia Analysis and Interation Lab (M4D), ITI-CERTH.

**Project title:** Satellite-Based Maritime Security Intelligence: Dark Vessel Detection, Oil Spill Segmentation, and AIS Cross-Verification Using Sentinel-1 SAR

**Abstract:**
We are developing a multi-module pipeline for automated maritime surveillance using Sentinel-1 SAR imagery. The system integrates (M1) SAR ship detection with oriented bounding boxes (HRSID, YOLO-family models), (M3) oil spill segmentation using transformer-based models (SegFormer), and (M2) AIS cross-referencing to flag dark/non-cooperative vessels. Current training data for the oil spill module comes from the Zenodo Sentinel-1 binary dataset (records 8346860/8253899/13761290). The Krestenitis/M4D dataset's five-class annotation (sea, oil spill, look-alike, ship, land) would allow us to directly model the look-alike class — which is the dominant source of false positives in our current binary classifier — and to evaluate cross-dataset generalisation between the two annotation schemes. Results will be published as part of a [MSc/BSc] thesis and a peer-reviewed journal/conference paper, with full attribution to your dataset.

**Supervisee:** [STUDENT NAME], [MSc/BSc student], [INSTITUTION], [STUDENT EMAIL]

We confirm that we have reviewed the Terms of Use and agree to comply fully. All use will be restricted to the research described above, and the dataset will not be redistributed.

Please let us know if you require any additional information.

Yours sincerely,

[SUPERVISOR NAME]
[TITLE]
[DEPARTMENT]
[INSTITUTION]
[INSTITUTIONAL EMAIL]
[PHONE — optional]

---

## What to fill in before sending

| Placeholder | What to put |
|---|---|
| `[SUPERVISOR NAME]` | Your supervisor's full name |
| `[TITLE]` | e.g. Associate Professor, Dr., Prof. |
| `[INSTITUTION]` | Full university name |
| `[STUDENT NAME]` | Your full name (Shaun Marvell Rodrigues) |
| `[MSc/BSc student]` | Whichever applies |
| `[STUDENT EMAIL]` | shaunmarvellrodrigues@gmail.com (or institutional if you have one) |
| `[MSc/BSc thesis]` | Match to your programme |

## Notes
- Do **not** send from a personal Gmail — it will be rejected. Supervisor's .edu/.ac.uk/institutional address only.
- The abstract above is already tailored to explain *why* the 5-class labels (especially look-alike) are needed — this is the strongest justification.
- If supervisor prefers a shorter email, the abstract can be shortened to the first two sentences + the look-alike rationale.
- Expect 1–2 week response time based on other researchers' experience.
