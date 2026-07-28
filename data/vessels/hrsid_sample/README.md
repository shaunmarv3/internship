# HRSID sample — 50 chips

A small, checked-in slice of HRSID so the detection code can be run without a 1.3 GB
download. **This is not the training set** — the reported mAP@50 = 0.938 comes from the
full 3,642 / 1,962 split, not from these 50 chips.

- 50 image/label pairs, one chip from each of 50 distinct HRSID source scenes
  (`P0001`…`P0137`), chosen for scene diversity
- every chip contains at least one ship, so labels are never empty
- 800×800 JPEG, YOLO OBB label format, single class `ship`
- ~6.6 MB total

## Full dataset

```bash
python download_vessel_data.py        # fetches the complete HRSID set into data/vessels/
```

Original: Wei et al., "HRSID: A High-Resolution SAR Images Dataset for Ship Detection and
Instance Segmentation", *IEEE Access* 8 (2020), 120234–120254.
