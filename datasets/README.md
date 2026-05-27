# Datasets

`datasets/` contains dataset interfaces and lightweight manifests. Large raw
data is not tracked in git.

Recommended local layout:

```text
datasets/
├── raw/
│   └── HanSig/
│       └── ... image files ...
├── processed/
│   └── ... generated preprocessing outputs ...
├── manifests/
└── interfaces/
```

Use `raw/` for immutable source data and `processed/` for generated outputs
such as JPEG conversions. Both folders are ignored by git except for `.gitkeep`
placeholders.

For `annotation_v2`, use the preparation script to copy single-signature JPGs
into `processed/csw_0116/{true,fake}/` and write manifests with
`image_id,image_path,label,authenticity`.

HanSig filenames are parsed from names like:

```text
original_w1_2_3.jpg
forgery_w1_2_3.jpg
```

The interface treats `w1_2` as the signer identity, where `w1` is the writer and
`2` is the signed-name index.
