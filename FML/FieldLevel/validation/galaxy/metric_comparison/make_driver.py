#!/usr/bin/env python3
"""Generate a narrowly hooked validation driver; preserve the original frozen source."""
from pathlib import Path
import argparse

def generate(source):
 changes=[('        int budget = 16;',
           '        GN_FIELD::load(root.parent_path() / ("field-metric-ensemble" + std::to_string(ensemble) + ".txt"));\n        int budget = 16;'),
          ('auto counted = FIELD_HMC::counted(target, s.white, s.parameters, frozen, o);',
           'auto counted = GN_FIELD::counted(target, s.white, s.parameters, frozen, o);'),
          ('        if (ensemble >= Ensembles ||',
           '        if (!field_settings.enabled) throw std::invalid_argument("Frozen-metric driver requires explicit field settings");\n        if (ensemble >= Ensembles ||'),
          ('return CP::identity(base, problem);',
           'return CP::identity(base, problem + ":frozen_field_metric=" + GN_FIELD::identity());')]
 for before,after in changes:
  if source.count(before)!=1:raise ValueError('Original driver hook missing or ambiguous: '+before)
  source=source.replace(before,after)
 return '#include <FML/FieldLevel/validation/galaxy/metric_comparison/Native.h>\n'+source

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 if a.source.resolve()==a.output.resolve():p.error('Never overwrite the original source')
 a.output.write_text(generate(a.source.read_text()))
if __name__=='__main__':main()
