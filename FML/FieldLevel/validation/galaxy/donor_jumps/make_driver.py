#!/usr/bin/env python3
"""Generate donor hooks on the exact-support driver without editing frozen sources."""
from pathlib import Path
import argparse,importlib.util
SUPPORT=Path(__file__).resolve().parent.parent/'support_interweave/make_driver.py'
COUNTERS=['donor_attempts','donor_accepted','donor_support_rejections','donor_values_total','donor_values_sampling','donor_large_attempts','donor_large_accepted']
def generate(source,support_path=SUPPORT):
 spec=importlib.util.spec_from_file_location('frozen_support_codegen',support_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);source=module.generate(source)
 changes=[('struct Stats {','struct Stats {\n    uint64_t '+', '.join(n+'=0' for n in COUNTERS)+';'),
 ('s.prior_refreshes};','s.prior_refreshes, '+', '.join('s.'+n for n in COUNTERS)+'};'),
 ('        v.prior_refreshes = d.integer();','        v.prior_refreshes = d.integer();\n'+''.join('        v.'+n+' = d.integer();\n' for n in COUNTERS)),
 ('        if (!valid)','        valid &= DONOR_DRIVER::valid_counters(v, uint64_t(s.next), draws, j + 1 < betas.size());\n        if (!valid)'),
 ('        ACTIVE_SUPPORT::InitialSupport<Model> support(model);','        ACTIVE_SUPPORT::InitialSupport<Model> support(model);\n        DONOR_DRIVER::prepare(support);'),
 ('        SUPPORT_DRIVER::load(root.parent_path() / "rescaling-policy.txt");','        SUPPORT_DRIVER::load(root.parent_path() / "rescaling-policy.txt");\n        DONOR_DRIVER::load(root.parent_path() / "donor-pool.txt");'),
 ('SUPPORT_DRIVER::identity(support));','SUPPORT_DRIVER::identity(support) + ":fixed_donors=" + DONOR_DRIVER::identity());'),
 ('    if (measured) {\n        stats.accept_sum += field.acceptance;','    DONOR_DRIVER::step(support, target, s, stats, current, work, seed, measured);\n    if (measured) {\n        stats.accept_sum += field.acceptance;')]
 for before,after in changes:
  if source.count(before)!=1:raise ValueError('Missing/ambiguous donor hook: '+before)
  source=source.replace(before,after)
 source=source.replace('"FML serial replica ensemble 2"','"FML serial fixed donor ensemble 1"')
 anchor="        out << '}';"
 if source.count(anchor)!=1:raise ValueError('Ambiguous summary hook')
 extra='        if (DONOR_DRIVER::enabled) out'+''.join(' << ",\\"'+n+'\\":" << v.'+n for n in COUNTERS)+';\n'
 source=source.replace(anchor,extra+anchor)
 return '#include <FML/FieldLevel/validation/galaxy/donor_jumps/Native.h>\n'+source

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--support-generator',type=Path,default=SUPPORT);a=p.parse_args()
 if a.source.resolve()==a.output.resolve():p.error('Never overwrite original source')
 a.output.write_text(generate(a.source.read_text(),a.support_generator))
if __name__=='__main__':main()
