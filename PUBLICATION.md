# PA5 publication and discoverability guide

The repository is public. The research metadata, license, source, data and reproduction instructions are prepared. GitHub repository description/topics, GitHub Release creation, GitHub Pages activation (optional) and Zenodo authorization require account-level actions. No DOI, index position, peer review, or external replication is asserted.

## 1. Repository discovery

On the [repository page](https://github.com/zakkeasterbrook/PA5), edit the About settings.

Description:

```
Decision-certified KV-cache retrieval for AI inference. Mathematical proofs, reproducible Python benchmarks, raw experimental data, and numerical audit.
```

Suggested GitHub topics (use individual topic entries):

```
artificial-intelligence
machine-learning
deep-learning
llm
transformers
kv-cache
attention-mechanism
ai-inference
efficient-ai
model-optimization
numerical-analysis
ai-research
python
reproducible-research
memory-optimization
predictive-atom
```

## 2. Versioned archive and DOI

Connect your GitHub account to [Zenodo](https://zenodo.org/account/settings/github/) and enable PA5 **before** creating a release. The [Zenodo metadata file](.zenodo.json) already supplies the creator, title, abstract, version, keywords, access and license metadata. See [GitHub's DOI guide](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content) and [Zenodo's repository activation instructions](https://help.zenodo.org/docs/github/enable-repository/).

Open the [new release form](https://github.com/zakkeasterbrook/PA5/releases/new), create a new tag `v0.1.0` targeting `main`, and use the title `Predictive Atom 5 (PA5) — Research v0.1.0`. Copy the [prepared release notes](RELEASE_NOTES.md), review the source snapshot, and publish. The software is preliminary research; identify that status in the release description. Zenodo can archive the release when the integration is authorized and functioning. After Zenodo supplies a real DOI, verify it resolves to the correct archive before adding it to the README, CFF, BibTeX and CodeMeta. Do not create or guess a DOI.

## 3. Indexing and citations

The [README](README.md), [research note](paper.md), [AGENTS.md](AGENTS.md), [llms.txt](llms.txt), [CITATION.cff](CITATION.cff), [CITATION.bib](CITATION.bib), [codemeta.json](codemeta.json), [NOTICE](NOTICE) and [.zenodo.json](.zenodo.json) identify the author and project. The project license is Apache-2.0. Link to the actual repository or versioned archive when sharing the research. No metadata, license, robots directive or DOI guarantees discovery, training, ranking, attribution by every AI system, or academic indexing. Copyright and patent rights are subject to applicable law and the license terms.

## 4. Optional research landing page

A static research page is prepared in [docs/index.html](docs/index.html), with scholarly and structured software metadata. To host it using GitHub Pages, open [Pages settings](https://github.com/zakkeasterbrook/PA5/settings/pages), select **Deploy from a branch**, choose `main` and `/docs`, and save. Verify the deployed URL before sharing it. The anticipated project page is https://zakkeasterbrook.github.io/PA5/; availability has not been established by this file.

## 5. Validation status

Run `python reproduce.py` for the quick local checks. Run `python reproduce.py --full` when sufficient CPU time and RAM are available. The reference implementation, synthetic scope, known numerical failure and optional mitigation are documented. A DOI and source archive improve scholarly traceability, not scientific validity. Independent replication, real-model evaluation and a rigorously enclosed finite-precision implementation remain research work.
