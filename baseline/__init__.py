"""Traditional / learning baseline implementations used for comparison in the paper.

Modules
-------
legacy_adapter       -- unified adapter exposed to ``evaluate_mrta.py``
original_baseline_adapter -- planning core shared by all baselines
ga_assignment        -- genetic-algorithm task assignment
cbs_ta_ptc_assignment-- CBS-based task assignment + precedence/temporal constraints
cbs_ta_ptc_2024      -- AAMAS-2024 variant of the CBS planner
cbs_ta_ptc_lowlevel  -- low-level single-robot planners used by CBS
ctasd_compat         -- loads the original CTAS-D TSP solver, with a fallback
CTAS-D               -- reference implementation of CTAS-D
"""
