# Datasets

`datasets/` contains dataset interfaces and lightweight manifests. Large raw
data is not tracked in git.

Recommended local layout:

```text
datasets/
├── raw/
│   └── HanSig/
│       └── ... image files ...
├── manifests/
└── interfaces/
```

HanSig filenames are parsed from names like:

```text
original_w1_2_3.jpg
forgery_w1_2_3.jpg
```

The interface treats `w1_2` as the signer identity, where `w1` is the writer and
`2` is the signed-name index.
