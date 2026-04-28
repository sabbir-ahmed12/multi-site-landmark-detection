# PRD — 

This repository prepares the **MIATT (ECE:5490 / 5940) final-exam student submitted solution **
A synthesized multi-site brain-MRI corpus derived from the
read-only sources, used to evaluate students on
site-agnostic anatomical landmark detection and ACPC alignment.

## Breadcrumbs

Where each piece of the project lives:

| Audience | Location | Contents |
|---|---|---|
| Course-wide | `README.md` (project root) | Course themes, lab environment, prompt library, lecture schedule. |
| Student (exam dataset) | `copy_miatt_shared_data/MIATTFINALEXAMDATA/` (deployed at `/nfs/s-l028/scratch/opt/ece5490/MIATTFINALEXAMDATA/`) | Six labeled `siteA`–`siteF` directories plus six held-out `site*_unlabeled` directories. |

## Data enclave policy

`MIATTFINALEXAMDATA/` must remain on the engineering computers.
Only the predicted landmark `.fcsv` files produced by students may be
shared publicly on GitHub; no other file derived from the enclave may
leave it.
