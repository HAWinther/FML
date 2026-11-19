#!/usr/bin/env python3
"""Generate only the exact-support rescaling hooks; preserve historical sources."""
from pathlib import Path
import argparse

def generate(source):
 changes=[('static void positive_update(const Model & model,','static void positive_update(ACTIVE_SUPPORT::InitialSupport<Model> & support, const Model & model,'),
 ('positive_update(model,','positive_update(support, model,'),
 ('        auto work = model.workspace();','        auto work = model.workspace();\n        ACTIVE_SUPPORT::InitialSupport<Model> support(model);'),
 ('auto proposal = RESCALING::propose(s.white,','auto proposal = SUPPORT_DRIVER::propose(support, s.white,'),
 ('        int budget = 16;','        SUPPORT_DRIVER::load(root.parent_path() / "rescaling-policy.txt");\n        int budget = 16;'),
 ('        if (ensemble >= Ensembles ||','        if (!field_settings.enabled) throw std::invalid_argument("Support driver requires explicit field settings");\n        if (ensemble >= Ensembles ||'),
 ('return CP::identity(base, problem);','return CP::identity(base, problem + ":rescaling_policy=" + SUPPORT_DRIVER::identity(support));')]
 for before,after in changes:
  if source.count(before)!=1:raise ValueError('Missing/ambiguous source hook: '+before)
  source=source.replace(before,after)
 return '#include <FML/FieldLevel/validation/galaxy/support_interweave/Native.h>\n'+source

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 if a.source.resolve()==a.output.resolve():p.error('Never overwrite original source')
 a.output.write_text(generate(a.source.read_text()))
if __name__=='__main__':main()
