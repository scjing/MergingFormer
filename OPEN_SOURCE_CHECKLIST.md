# Open-source release checklist

Use this checklist before making the repository public.

## Ownership and licensing

- [x] Select an open-source license and add the complete license text as
      `LICENSE`.
- [ ] Confirm who owns the copyright: the author, an employer, a university, or
      another institution.
- [ ] Identify code adapted or copied from papers, repositories, notebooks, and
      course material.
- [ ] Confirm that every third-party license permits redistribution under the
      selected project license.
- [ ] Retain required copyright notices and add source links and modification
      notes where applicable.
- [ ] Obtain agreement from all contributors whose work is included.

## Data, models, and privacy

- [x] Record that the trajectory dataset is not open-sourced or distributed and
      that users must supply their own compatible data.
- [ ] Check CSV files, plots, logs, and checkpoints for personal, confidential,
      licensed, or identifying information.
- [ ] Publish only data samples whose license and privacy terms allow it.
- [ ] Record the source, license, units, sampling interval, and preprocessing
      steps for every published dataset.
- [ ] Confirm that pretrained model weights may be distributed.

## Secrets and generated files

- [x] Search the current source tree for passwords, API keys, tokens,
      private URLs, local usernames, and machine-specific paths.
- [ ] Repeat the secrets check for Git history after the first commit and before
      publication.
- [x] Confirm `.gitignore` excludes datasets, checkpoints, virtual environments,
      IDE settings, caches, and experiment outputs.
- [x] Inspect the staged file list and the size of every file before the first
      commit.

## Documentation and reproducibility

- [x] Add an initial project overview and directory map.
- [x] Document the current data schema and known inconsistencies.
- [x] Add an inferred dependency list with an unverified-version notice.
- [ ] Add one documented training command after validating it from a clean
      checkout.
- [ ] Add one documented evaluation command after validating checkpoint loading.
- [x] Record the original Python 3.9 environment from the available package
      screenshots in `environment.yml` and `requirements.txt`.
- [ ] Recreate and validate the recorded environment from a clean checkout.
- [ ] Explain the model architecture, expected inputs and outputs, metrics, and
      reported results.
- [x] Add manuscript title and authors to `CITATION.cff`.
- [ ] Add the publication venue, year, DOI, and paper URL when finalized.

## Repository maintenance

- [ ] Choose an issue tracker and define what information bug reports require.
- [ ] Configure repository topics, description, and social preview image.
- [ ] Enable branch protection and automated checks when a test workflow exists.
- [ ] Clone the public repository into a new directory and follow the README as
      a final release check.
- [ ] Create the first tagged release and describe its known limitations.
