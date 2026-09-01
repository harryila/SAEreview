# Third-party data and models

The MIT license in this repository applies only to original SAEreview code and
documentation. It does not relicense third-party data, model weights, papers, or
generated assets.

## SCAR

SCAR is maintained at <https://github.com/siyuyuan/scar>. At pinned commit
`3dfc897cf6cc685531edc80ab64f35660403fc6c`, the repository contains citation and
research-use language but no explicit redistribution license. Therefore `main`
does not bundle the dataset. `scripts/prepare_retrieval_assets.py` downloads the
exact upstream file for the user and verifies SHA-256
`12883db11de17454b3a4ae30a109f4b64861125b1e94846e17b8edc3f8a12369`.

## Scientific embedding SAEs

The cs.LG and astro.PH checkpoints are downloaded from the authors' public
Hugging Face dataset at pinned revision
`b2cbb184b58880b77a546511e11d8fd214c40556`. That artifact repository has no
dataset card or explicit license metadata, so the files remain external and any
upstream terms continue to apply. Each exact k64/n9216 checkpoint also exceeds
GitHub's ordinary 100 MB file limit.

## Gemma 2 and Gemma Scope

The Latent Escape protocol uses `google/gemma-2-9b-it`, which is gated and subject
to the Gemma Terms of Use, and `google/gemma-scope-9b-it-res`, whose Hugging Face
model card declares CC BY 4.0. Users must accept and follow the applicable upstream
terms. This repository does not redistribute either set of weights.
