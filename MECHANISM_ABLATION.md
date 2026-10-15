# SAG-AlignNet mechanism variants

The optional `--mechanism_variant` argument exposes alternative prototype and masked-alignment components for controlled experiments. The default `native` selects the checked-in SAG-AlignNet configuration used by the bundled checkpoints.

```bash
--mechanism_variant native
--mechanism_variant cprod=plot
--mechanism_variant cprod=prior
--mechanism_variant masalign=irra
--mechanism_variant masalign=rasa
--mechanism_variant cprod=plot,masalign=irra
```

`cprod=plot` uses iterative slot attention; `cprod=prior` uses a learnable codebook and quantization/reconstruction loss. `masalign=irra` uses the shared cross-modal fusion and standard MLM objective; `masalign=rasa` adds in-batch matching, positive-relation, and replaced-token objectives. CProD alternatives require `sdm` in `loss_names`; MasAlign alternatives require `mlm`. Each alternative changes model components and must be trained from scratch with a separate output directory.

The `rasa` option adapts the objectives to this repository's annotation interface: it does not use a momentum teacher, and the same-PID replacement pairs available in a batch determine PRD supervision. Treat it as a repository-specific adaptation rather than an exact reproduction of another method.

The variants enable controlled comparisons but are not evidence of improved performance by themselves. Report measured results under matched data splits, seeds, and training budgets.
