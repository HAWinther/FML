# Next stage: run the frozen prior-predictive pilot

[Prior sensitivity](../../../docs/validation/studies/prior_sensitivity/RESULTS.md),
[prior-predictive truth controls](../../../docs/validation/studies/prior_predictive_controls/RESULTS.md)
and [calibration runner controls](../../../docs/validation/studies/calibration_runner/RESULTS.md)
are complete. The study runner monitors likelihood Q alongside the existing
eleven coordinates, retains failed/incomplete outcomes, and supports checkpoint
pause/resume with a conservative cumulative time ledger.

Next: launch the separately frozen eight-dataset pilot (IDs 201–208), four chains
per dataset, 256 warmup +2048 retained draws, with the exact saved seeds, gates,
endpoint ranks and 16-hour sampling cap. It is not started. The raw frozen pilot
and launch/pause/resume/analysis scripts are under the external fieldlevel-calibration
output directory; see the [maintained study instructions](../calibration/README.md).
No replacement truths, omitted failures, cap increases or extension until pass.

Independent nonlinear calibration, exact challenge IC/spectrum/cosmology inputs,
larger-target validation and production MPI replica orchestration remain open.
