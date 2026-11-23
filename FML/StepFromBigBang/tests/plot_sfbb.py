"""Accuracy figures for StepFromBigBang (data from test_sfbb dump, EdS, 64^3, exact NUFFT force).
Usage: python3 plot_sfbb.py <plotdata dir> <output dir>"""
import os, sys, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D, OUT = sys.argv[1], sys.argv[2]
N = 64
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'magenta': '#e87ba4',
     'green': '#008300'}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False})

def load(case, name):
    return np.fromfile(os.path.join(D, case, name + '.bin'), dtype=np.float64).reshape(N, N, N, 3)

def rel(a, b, z):
    return np.sqrt(np.mean((a - b) ** 2) / np.mean((b - z) ** 2))

kf = np.fft.fftfreq(N) * N
kr = np.fft.rfftfreq(N) * N
kk = np.sqrt(kf[:, None, None] ** 2 + kf[None, :, None] ** 2 + kr[None, None, :] ** 2) / (N / 2)  # k / k_Ny
edges = np.geomspace(1.0 / 32, np.sqrt(3), 17)
kc = np.sqrt(edges[1:] * edges[:-1])
ib = np.digitize(kk.ravel(), edges) - 1
def binsum(x):
    return np.bincount(ib[(ib >= 0) & (ib < len(kc))], weights=x.ravel()[(ib >= 0) & (ib < len(kc))], minlength=len(kc))
def ft(f):
    return [np.fft.rfftn(f[..., i]) for i in range(3)]
def power(f):
    F = ft(f); return binsum(sum(np.abs(x) ** 2 for x in F))
def cross(f, g):
    F, G = ft(f), ft(g); return binsum(sum((x * np.conj(y)).real for x, y in zip(F, G)))

cases = [('k0.25', s) for s in ('0.1', '0.3', '1.0')] + [('k1.0', s) for s in ('0.1', '0.3', '1.0')]
label = {'k0.25': 'smooth field (k < k$_{Ny}$/4)', 'k1.0': 'full band'}
BF, L3 = [1, 2, 4, 8, 16], [1, 2, 4, 8]
summary = {}

# ---------------------------------------------------------------- (a) convergence
fig, axs = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True)
for ax, (kc_, s) in zip(axs.ravel(), cases):
    case = f'{kc_}_s{s}'
    z, r, r2 = load(case, 'za'), load(case, 'bf64'), load(case, 'l3_32')
    eb = [rel(load(case, f'bf{n}'), r, z) for n in BF]
    el = [rel(load(case, f'l3_{n}'), r, z) for n in L3]
    e2, e3, floor = rel(load(case, 'lpt2'), r, z), rel(load(case, 'lpt3'), r, z), rel(r2, r, z)
    summary[case] = dict(bullfrog=eb, lpt3=el, ic_2lpt=e2, ic_3lpt=e3, ref_floor=floor)
    ax.axhspan(1e-9, floor, color=GRID, alpha=0.7, lw=0)
    ax.plot(BF, eb, '-o', color=C['blue'], lw=2, ms=6, mec=SURF, mew=1.2, label='BullFrog (1 force / step)')
    ax.plot([3 if n == 1 else 2 * n for n in L3], el, '-s', color=C['orange'], lw=2, ms=6, mec=SURF, mew=1.2,
            label='lpt3 (3 forces for 1 step, 2n for n steps)')
    if abs(np.log10(e2 / e3)) < 0.12:   # the two IC lines nearly coincide: one label
        ax.axhline(e2, color=INK2, lw=1, ls='--')
        ax.text(17.5, max(e2, e3), '2LPT and 3LPT IC', color=INK2, fontsize=8, va='bottom', ha='right')
    else:
        for e, name, ls in ((e2, '2LPT IC', '--'), (e3, '3LPT IC', ':')):
            ax.axhline(e, color=INK2, lw=1, ls=ls)
            ax.text(17.5, e, name, color=INK2, fontsize=8, va='bottom', ha='right')
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(['1', '2', '4', '8', '16'])
    ax.set_ylim(max(floor / 3, 1e-6), None)
    ax.set_title(f'{label[kc_]}, $\\sigma_\\delta$ = {s}', fontsize=9.5, color=INK)
for ax in axs[1]: ax.set_xlabel('force evaluations')
for ax in axs[:, 0]: ax.set_ylabel("position error / beyond-Zel'dovich")
axs[0, 0].legend(fontsize=8.5, loc='lower left')
fig.suptitle("StepFromBigBang: error from D = 0 against 64 BullFrog steps (EdS, $64^3$, exact force; "
             "grey: reference uncertainty, 64 BullFrog vs 32 lpt3 steps)", fontsize=9.5, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'sfbb_convergence.png'), dpi=150); plt.close(fig)

# ---------------------------------------------------------------- (b), (c) scale dependence
case = 'k1.0_s0.3'
z, r = load(case, 'za'), load(case, 'bf64')
Pb = power(r - z)
def ek(name):
    return np.sqrt(power(load(case, name) - r) / Pb)
fig, axs = plt.subplots(3, 2, figsize=(11, 11.4))
ax = axs[0, 0]
for (name, lab, col, mk, ls) in [('lpt2', '2LPT IC', C['magenta'], 'v', '--'), ('lpt3', '3LPT IC', C['green'], '^', ':'),
                                 ('bf1', 'BullFrog, 1 step', C['blue'], 'o', '-'), ('bf2', 'BullFrog, 2 steps', C['aqua'], 'D', '-'),
                                 ('l3_1', 'lpt3, 1 step (3 forces)', C['orange'], 's', '-'),
                                 ('l3_2', 'lpt3, 2 steps (4 forces)', C['yellow'], 'P', '-')]:
    ax.loglog(kc, ek(name), ls=ls, marker=mk, color=col, lw=1.8, ms=5, mec=SURF, mew=1, label=lab)
ax.set_title('error per k: methods (full band, $\\sigma_\\delta$ = 0.3)', fontsize=9.5)
ax.legend(fontsize=8)
ax = axs[0, 1]
FV = [('bf1', 'exact (NUFFT)', C['blue'], 'o', '-'),
      ('bf1_sheet_PCS_2x_il2', 'sheet, PCS 2x, interlaced', C['aqua'], 'D', '-'),
      ('bf1_sheet_PCS_1x', 'sheet, PCS 1x', C['green'], '^', '-'),
      ('bf1_pm_PCS_1x_dc', 'PM, PCS 1x, deconvolved', C['orange'], 's', '--'),
      ('bf1_pm_CIC_1x', 'PM, CIC 1x', C['yellow'], 'P', '--'),
      ('bf1_pm_CIC_2x', 'PM, CIC 2x', C['magenta'], 'v', '--')]
for (name, lab, col, mk, ls) in FV:
    ax.loglog(kc, ek(name), ls=ls, marker=mk, color=col, lw=1.8, ms=5, mec=SURF, mew=1, label=lab)
summary['one_step_force_variants'] = {n: rel(load(case, n), r, z) for n in
                                      [f[0] for f in FV] + ['bf1_sheet_PCS_2x', 'bf1_pm_PCS_1x_dc_il2', 'bf1_pm_PCS_2x_dc', 'bf1_pm_PCS_2x_dc_il2', 'bf1_sheet_nufft']}
ax.set_title('error per k: one BullFrog step with each force', fontsize=9.5)
ax.legend(fontsize=7.8, loc='upper left', ncol=2)
ax.set_ylim(None, 60)
ax.text(0.98, 0.03, "reference: the exact particle force; the sheet\nremoves the lattice's discreteness near k$_{Ny}$ by design",
        transform=ax.transAxes, ha='right', va='bottom', fontsize=7.8, color=INK2)
for ax in axs[0]:
    ax.set_xlabel('k / k$_{Ny}$'); ax.set_ylabel("error / beyond-Zel'dovich (per k)")
    ax.axvline(1, color=INK2, lw=0.8)

# (c) the forces: |T - 1| and 1 - r against the exact force, near the lattice (Zel'dovich at D = 1/2, sigma = 0.3)
# and late (16 BullFrog steps to D = 1, sigma = 1, past shell crossing), with the nonlinear part of the exact force
FF = [('sheet_PCS_1x', 'sheet, PCS 1x', C['green'], '^', '-'), ('sheet_PCS_2x_il2', 'sheet, PCS 2x, interlaced', C['aqua'], 'D', '-'),
      ('pm_CIC_1x', 'PM, CIC 1x', C['yellow'], 'P', '--'), ('pm_PCS_1x_dc', 'PM, PCS 1x, deconvolved', C['orange'], 's', '--'),
      ('pm_PCS_1x_dc_il2', 'PM, PCS 1x, deconv., interlaced', C['blue'], 'o', '--'), ('pm_CIC_2x', 'PM, CIC 2x', C['magenta'], 'v', '--')]
summary['force_rms_difference'] = {}
summary['force_rms_difference_late'] = {}
k0 = lambda a: 1.5 * np.sqrt(a)
for col_i, (fcase, prefix, Dval, title) in enumerate([
        ('k1.0_s0.3', 'force_', 0.5, 'near the lattice: Zel\'dovich at D = 1/2, $\\sigma_\\delta$ = 0.3'),
        ('k1.0_s1.0', 'lateforce_', 1.0, 'late: after 16 steps to D = 1, $\\sigma_\\delta$ = 1 (shell crossing)')]):
    Fx = load(fcase, prefix + 'exact')
    Pex = power(Fx)
    Flin = k0(Dval) * Dval * load(fcase, 'za')           # the linear force D Psi1 (times kappa0)
    nonlin = np.sqrt(power(Fx - Flin) / Pex)
    for row in (1, 2):
        ax = axs[row, col_i]
        if row == 1:
            ax.axhspan(1e-9, 1e-2, color=GRID, alpha=0.7, lw=0)
            ax.text(0.035, 1.2e-2, '1% of the force', fontsize=7.8, color=INK2, va='bottom')
            ax.loglog(kc, nonlin, '-', color=INK, lw=2.4, label='nonlinear part of the exact force')
        for (name, lab, col, mk, ls) in FF:
            Fm = load(fcase, prefix + name)
            X, Pm = cross(Fm, Fx), power(Fm)
            y = np.abs(X / Pex - 1) if row == 1 else 1 - X / np.sqrt(Pm * Pex)
            ax.loglog(kc, np.maximum(y, 1e-9), ls=ls, marker=mk, color=col, lw=1.6, ms=4.5, mec=SURF, mew=1, label=lab)
            key = 'force_rms_difference' + ('' if col_i == 0 else '_late')
            summary[key][name] = float(np.sqrt(np.mean((Fm - Fx) ** 2) / np.mean(Fx ** 2)))
        ax.axvline(1, color=INK2, lw=0.8)
        ax.set_xlabel('k / k$_{Ny}$')
        ax.set_ylim(1e-9 if row == 2 else 1e-5, 10)
        ax.set_title(('|transfer - 1|\n' if row == 1 else 'decorrelation 1 - r(k)\n') + title, fontsize=9)
    axs[1, col_i].set_ylabel('|<F F$_{exact}$>/<F$_{exact}^2$> - 1|')
    note = ("full deconvolution on a 2x grid amplifies the lattice's\ndiscreteness here (PCS 2x: force 50% too large; not shown)"
            if col_i == 0 else "the exact force is cut at k$_{Ny}$; 2x grids also resolve\nthe clustered small scales. The sheet assumes a single stream")
    axs[2, col_i].text(0.03, 0.97, note, transform=axs[2, col_i].transAxes, fontsize=7.6, color=INK2, va='top')
    axs[2, col_i].set_ylabel('1 - r(k)')
axs[1, 0].legend(fontsize=7.5, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'sfbb_scale_dependence.png'), dpi=150); plt.close(fig)

# ---------------------------------------------------------------- (d) order
sig, eb1, el1 = [], [], []
for s in ('0.03', '0.1', '0.3', '1.0'):
    case = f'k0.25_s{s}'
    z, r = load(case, 'za'), load(case, 'l3_32')   # the higher-order reference at small amplitude
    sig.append(float(s)); eb1.append(rel(load(case, 'bf1'), r, z)); el1.append(rel(load(case, 'l3_1'), r, z))
sig, eb1, el1 = map(np.array, (sig, eb1, el1))
summary['order'] = dict(sigma=sig.tolist(), bullfrog_1=eb1.tolist(), lpt3_1=el1.tolist())
fig, ax = plt.subplots(figsize=(5.4, 4.1))
ax.loglog(sig, eb1, '-o', color=C['blue'], lw=2, ms=7, mec=SURF, mew=1.2, label='1 BullFrog step')
ax.loglog(sig, el1, '-s', color=C['orange'], lw=2, ms=7, mec=SURF, mew=1.2, label='1 lpt3 step')
g = np.array([0.025, 0.35])
gb, gl = 0.4 * eb1[0] * g / sig[0], 0.4 * el1[0] * (g / sig[0]) ** 2
ax.loglog(g, gb, ':', color=C['blue'], lw=1.3)
ax.loglog(g, gl, ':', color=C['orange'], lw=1.3)
ax.text(0.36, gb[-1], ' $\\propto\\sigma$', color=INK2, va='center')
ax.text(0.36, gl[-1], ' $\\propto\\sigma^2$', color=INK2, va='center')
ax.axvspan(0.6, 1.3, color=GRID, alpha=0.7, lw=0); ax.text(0.62, ax.get_ylim()[0] * 1.5, 'shell\ncrossing', color=INK2, fontsize=8)
ax.set_xlabel('$\\sigma_\\delta$ (linear, at the end)'); ax.set_ylabel("error / beyond-Zel'dovich")
ax.set_title('order: one step from D = 0, smooth field\n(reference: 32 lpt3 steps)', fontsize=9.5)
ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'sfbb_order.png'), dpi=150); plt.close(fig)

json.dump(summary, open(os.path.join(OUT, 'sfbb_summary.json'), 'w'), indent=1)
print(json.dumps(summary, indent=1)[:3000])
