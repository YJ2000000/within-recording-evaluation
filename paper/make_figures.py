"""Manuscript figures, R3 (journal black/gray style; scientific-schematic-figures conventions).

Drawn at final size (6.5 in wide), Helvetica-class font (TeX Gyre Heros), black/white/grays only, so every figure
survives grayscale printing. Series are told apart by marker shape and fill, never by colour alone. No in-figure
titles; panel letters a, b, c. Every number comes from a result file (paths below); nothing is typed by hand.
Outputs: figs/figN_*.pdf (vector, for the journal) and figs/figN_*.png (600 dpi, embedded in the Word draft).

Fig. 1  schematic: analysis pipeline (a), evidence structure (b), evaluation protocols and controls (c)
Fig. 2  engagement vs difficulty per dataset (a); task vs start rest and task vs end rest (b)
Fig. 3  heart-rate difficulty effect vs drift over one session, in bpm
Fig. 4  split-half null by signal (a), null vs training-test gap (b), protocols vs a time-only model (c)
Fig. 5  usable beats at rest and during tasks (a); wrist heart rate within 5 bpm of the reference (b)
Fig. 6  literature audit
"""
import json, os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
from matplotlib.lines import Line2D

mpl.rcParams.update({'font.family': 'TeX Gyre Heros', 'pdf.fonttype': 42, 'svg.fonttype': 'path', 'font.size': 7,
                     'axes.linewidth': 0.6, 'axes.edgecolor': 'black', 'axes.labelsize': 7, 'xtick.labelsize': 6.5,
                     'ytick.labelsize': 6.5, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'legend.fontsize': 6.3, 'lines.solid_capstyle': 'butt', 'savefig.dpi': 600})
R = '/path/to/project/data/results'; R3 = f'{R}/r3'; D = '/path/to/project/paper/data'; FIG = '/path/to/project/paper/figs'
INK, GRAY, LIGHT = 'black', '0.35', '0.86'
W = 6.5
RNG = np.random.default_rng(0)
SHRUNK = []
# one encoding for three series, reused across figures: filled black circle / gray square / open diamond
S1 = dict(marker='o', s=22, facecolor='black', edgecolor='black', linewidth=0.6)
S2 = dict(marker='s', s=20, facecolor='0.6', edgecolor='black', linewidth=0.6)
S3 = dict(marker='D', s=18, facecolor='white', edgecolor='black', linewidth=0.8)
S4 = dict(marker='^', s=22, facecolor='0.6', edgecolor='black', linewidth=0.6)     # EDA (Fig. 4)
SC = dict(marker='o', s=22, facecolor='0.6', edgecolor='black', linewidth=0.6)     # second task-vs-rest series


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIG}/{name}.{ext}', facecolor='white')
    plt.close(fig)


def letter(ax_or_fig, x, y, s, fig=None):
    (fig or ax_or_fig).text(x, y, s, fontsize=8.5, fontweight='bold', va='top', ha='left',
                            transform=(fig or ax_or_fig).transFigure if fig is not None else ax_or_fig.transAxes)


def boot(v, n=4000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    m = [RNG.choice(v, len(v)).mean() for _ in range(n)]
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def point(ax, x, y, lo, hi, style, label=None, horizontal=False):
    if horizontal:
        ax.plot([lo, hi], [y, y], color=INK, lw=0.9, zorder=3)
        ax.scatter([x], [y], zorder=4, label=label, **style)
    else:
        ax.plot([x, x], [lo, hi], color=INK, lw=0.9, zorder=3)
        ax.scatter([x], [y], zorder=4, label=label, **style)


def chance(ax, v=0.5, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(v, color=GRAY, lw=0.6, ls=(0, (3, 2)), zorder=1)


# =============================================================== Figure 1 (schematic)
def fig1():
    H = 5.56
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    REND = fig.canvas.get_renderer()

    def text(x, y, s, **kw):
        kw.setdefault('va', 'center'); kw.setdefault('ha', 'center'); kw.setdefault('zorder', 6)
        return ax.text(x, y, s, **kw)

    def fit(t, maxw):
        f0 = t.get_fontsize()
        while t.get_window_extent(renderer=REND).width / fig.dpi > maxw and t.get_fontsize() > 5.5:
            t.set_fontsize(t.get_fontsize() - 0.2)
        if t.get_fontsize() < f0:
            SHRUNK.append(('fig1', t.get_text()[:30], f0, round(t.get_fontsize(), 1)))
        return t

    def rect(x, y, w, h, fc='white', ec=INK, lw=0.6, z=3, hatch=None):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, hatch=hatch))

    def block_arrow(x0, x1, yc, h=0.16):
        w = x1 - x0; hw = h * 0.45
        pts = [(x0, yc - hw / 2), (x1 - 0.06, yc - hw / 2), (x1 - 0.06, yc - h / 2), (x1, yc), (x1 - 0.06, yc + h / 2),
               (x1 - 0.06, yc + hw / 2), (x0, yc + hw / 2)]
        ax.add_patch(Polygon(pts, closed=True, fc='white', ec=INK, lw=0.6, zorder=4))

    # ---- panel a: pipeline strip -------------------------------------------------------------
    top_a = H - 0.06
    letter(None, 0.04 / W, top_a / H, 'a', fig=fig)
    text(0.25, top_a - 0.07, 'Analysis pipeline (identical for all seven datasets)', ha='left', fontsize=7.2, fontweight='bold')
    stages = [('Data', '7 public datasets\n196 participants'), ('Windows', '30 s long\n10 s step'),
              ('Quality', '>= 80% valid beat\nintervals per\nwindow'),
              ('Normalise', 'z-score vs. the\nsession\'s rest\n(default)'), ('Features', 'cardiac, EDA,\ntemperature,\nrespiration, motion'),
              ('Models', 'logistic regr.;\nrandom forest;\nNormWear'), ('Evaluate', '6 protocols,\n5 controls,\nmixed models')]
    n = len(stages); gap = 0.15; x0 = 0.12; bw = (W - 2 * x0 - (n - 1) * gap) / n; bh = 0.66
    ytop = top_a - 0.22
    for i, (t1, t2) in enumerate(stages):
        x = x0 + i * (bw + gap)
        rect(x, ytop - bh, bw, bh, fc='white')
        rect(x, ytop - 0.17, bw, 0.17, fc='0.2', ec=INK)
        fit(text(x + bw / 2, ytop - 0.085, f'{i + 1}  {t1}', fontsize=6.8, fontweight='bold', color='white'), bw - 0.06)
        fit(text(x + bw / 2, ytop - 0.17 - (bh - 0.17) / 2, t2.replace('>=', '≥'), fontsize=5.9, color=INK, linespacing=1.15), bw - 0.06)
        if i < n - 1:
            block_arrow(x + bw + 0.015, x + bw + gap - 0.015, ytop - bh / 2)
    bottom_a = ytop - bh

    # ---- panel b: evidence structure -----------------------------------------------------------
    top_b = bottom_a - 0.16
    ax.add_line(Line2D([0.04, W - 0.04], [top_b + 0.07, top_b + 0.07], color='0.6', lw=0.4))
    letter(None, 0.04 / W, top_b / H, 'b', fig=fig)
    text(0.25, top_b - 0.07, 'Evidence structure', ha='left', fontsize=7.2, fontweight='bold')
    comps = [('Development (exploratory)', 'COG-BCI, UNIVERSE, MultiPhysio-HRC\nmechanisms found; formal tests on the same data (Table 3)', '0.93', INK),
             ('Plan fixed', 'claims and analyses\nwritten down before the\nheld-out data were analysed', '0.72', INK),
             ('Held-out test', 'WESAD, MAUS, ds007554, CogWear\nclaims judged by one CI rule (Table 4)', '0.25', 'white')]
    widths = [2.5, 1.2, 2.35]; gap_b = 0.2; xb = (W - sum(widths) - 2 * gap_b) / 2; hb = 0.52; yb = top_b - 0.2
    for k, ((t1, t2, fc, tc), wdt) in enumerate(zip(comps, widths)):
        rect(xb, yb - hb, wdt, hb, fc=fc, ec=INK)
        fit(text(xb + wdt / 2, yb - 0.11, t1, fontsize=6.8, fontweight='bold', color=tc), wdt - 0.08)
        fit(text(xb + wdt / 2, yb - 0.33, t2, fontsize=5.9, color=tc, linespacing=1.15), wdt - 0.08)
        if k < 2:
            block_arrow(xb + wdt + 0.02, xb + wdt + gap_b - 0.02, yb - hb / 2)
        xb += wdt + gap_b
    bottom_b = yb - hb

    # ---- panel c: protocols ----------------------------------------------------------------------
    top_c = bottom_b - 0.16
    ax.add_line(Line2D([0.04, W - 0.04], [top_c + 0.07, top_c + 0.07], color='0.6', lw=0.4))
    letter(None, 0.04 / W, top_c / H, 'c', fig=fig)
    text(0.25, top_c - 0.07, 'Evaluation protocols and controls', ha='left', fontsize=7.2, fontweight='bold')
    lx = W - 3.05
    for fc, lab in (('black', 'training windows'), ('0.6', 'test windows'), ('white', 'not used (gap, Fig. 3b)')):
        rect(lx, top_c - 0.115, 0.16, 0.09, fc=fc, lw=0.5)
        t_ = text(lx + 0.2, top_c - 0.07, lab, ha='left', fontsize=5.9, color=INK)
        lx += 0.2 + t_.get_window_extent(renderer=REND).width / fig.dpi + 0.18
    LX = 2.25                                   # timeline starts here; row labels to the left
    RW, TW, BG = 0.30, 0.50, 0.025              # rest width, task width, gap between blocks
    day1 = [('rest', RW), ('easy', TW), ('hard', TW), ('easy', TW), ('hard', TW)]
    day2 = [('rest', RW), ('hard', TW), ('easy', TW)]
    xs1, x = [], LX
    for nm, w in day1:
        xs1.append((nm, x, w)); x += w + BG
    D2 = x + 0.22
    xs2, x = [], D2
    for nm, w in day2:
        xs2.append((nm, x, w)); x += w + BG
    rh = 0.17; row_dy = 0.335
    rows = ['Recording', 'Population (LOSO)', 'Random windows', 'Same-session personal', 'Cross-block (same day)',
            'Leave-one-block-pair-out', 'Cross-session personal', 'Split-half null', 'Same-condition blocks', 'Clock baseline']
    desc = ['two sessions; order varies by person', 'train on other people; test this person', 'windows split at random within blocks',
            'train: first k windows per block; test: later ones', 'same training windows; test: blocks 3-4',
            'train: two blocks; test: the other two (all folds)', 'same training windows; test: day 2',
            'halves of ONE block labelled 0 / 1', 'two blocks of one condition labelled 0 / 1', 'the only feature is time in the session']
    fill = {'rest': 'white', 'easy': 'white', 'hard': 'white'}
    y0 = top_c - 0.36
    for r, (title, dsc) in enumerate(zip(rows, desc)):
        yc = y0 - r * row_dy
        text(0.25, yc + 0.055, title, ha='left', fontsize=6.6, fontweight='bold')
        fit(text(0.25, yc - 0.075, dsc, ha='left', fontsize=5.8, color=GRAY), LX - 0.35)
        if r == 0:
            for nm, xx, w in xs1 + xs2:
                rect(xx, yc - rh / 2, w, rh, fc=fill[nm], lw=0.6)
                text(xx + w / 2, yc, nm, fontsize=5.9)
            text((xs1[0][1] + xs1[-1][1] + xs1[-1][2]) / 2, yc + 0.17, 'day 1', fontsize=5.9, color=GRAY)
            text((xs2[0][1] + xs2[-1][1] + xs2[-1][2]) / 2, yc + 0.17, 'day 2', fontsize=5.9, color=GRAY)
            for k, (nm, xx, w) in enumerate(xs1[1:]):
                text(xx + w / 2, yc - 0.15, f'block {k + 1}', fontsize=6.0, color=GRAY)
            text(D2 - 0.11, yc, '//', fontsize=6.5, color=GRAY)
            continue
        tasks1 = [b for b in xs1 if b[0] != 'rest']; tasks2 = [b for b in xs2 if b[0] != 'rest']
        if title.startswith('Population'):
            for _, xx, w in tasks1 + tasks2:
                rect(xx, yc - rh / 2, w, rh, fc='0.6', lw=0.4)
        elif title.startswith('Random'):
            for _, xx, w in tasks1[:2]:
                nwin = 10
                for j in range(nwin):
                    rect(xx + j * w / nwin, yc - rh / 2, w / nwin, rh, fc='black' if RNG.random() < 0.5 else '0.6', lw=0.0, ec='none')
                rect(xx, yc - rh / 2, w, rh, fc='none', lw=0.4)
        elif title.startswith('Same-session'):
            for _, xx, w in tasks1[:2]:
                rect(xx, yc - rh / 2, 0.14, rh, fc='black', lw=0.4)
                rect(xx + 0.14, yc - rh / 2, 0.06, rh, fc='white', lw=0.4)
                rect(xx + 0.20, yc - rh / 2, w - 0.20, rh, fc='0.6', lw=0.4)
        elif title.startswith('Cross-block'):
            for _, xx, w in tasks1[:2]:
                rect(xx, yc - rh / 2, 0.14, rh, fc='black', lw=0.4)
                rect(xx + 0.14, yc - rh / 2, w - 0.14, rh, fc='white', lw=0.4)
            for _, xx, w in tasks1[2:]:
                rect(xx, yc - rh / 2, w, rh, fc='0.6', lw=0.4)
        elif title.startswith('Leave-one'):
            for k, (_, xx, w) in enumerate(tasks1):
                rect(xx, yc - rh / 2, w, rh, fc='black' if k in (0, 3) else '0.6', lw=0.4)
            text(tasks1[-1][1] + tasks1[-1][2] + 0.06, yc, 'one of four folds shown', ha='left', fontsize=5.8, color=GRAY)
        elif title.startswith('Cross-session'):
            for _, xx, w in tasks1[:2]:
                rect(xx, yc - rh / 2, 0.14, rh, fc='black', lw=0.4)
                rect(xx + 0.14, yc - rh / 2, w - 0.14, rh, fc='white', lw=0.4)
            for _, xx, w in tasks2:
                rect(xx, yc - rh / 2, w, rh, fc='0.6', lw=0.4)
        elif title.startswith('Split-half'):
            _, xx, w = tasks1[0]; h2 = w / 2
            for k, xh in enumerate((xx, xx + h2)):
                rect(xh, yc - rh / 2, 0.07, rh, fc='black', lw=0.4)
                rect(xh + 0.07, yc - rh / 2, 0.035, rh, fc='white', lw=0.4)
                rect(xh + 0.105, yc - rh / 2, h2 - 0.105, rh, fc='0.6', lw=0.4)
                text(xh + 0.105 + (h2 - 0.105) / 2, yc, f'{k}', fontsize=6.0, color=INK)
            ax.add_line(Line2D([xx + h2] * 2, [yc - rh / 2 - 0.03, yc + rh / 2 + 0.03], color=INK, lw=0.9, zorder=5))
            text(tasks1[1][1] + 0.02, yc, 'labels 0 / 1 differ only in time: AUROC above 0.5 = drift or slow autocorrelation', ha='left', fontsize=5.8, color=GRAY)
        elif title.startswith('Same-condition'):
            for k, (_, xx, w) in enumerate((tasks1[0], tasks1[2])):
                rect(xx, yc - rh / 2, 0.14, rh, fc='black', lw=0.4)
                rect(xx + 0.14, yc - rh / 2, w - 0.14, rh, fc='0.6', lw=0.4)
                text(xx + 0.14 + (w - 0.14) / 2, yc, f'{k}', fontsize=6.0, color=INK)
            text(tasks1[2][1] + tasks1[2][2] + 0.06, yc, 'both blocks "easy": AUROC above 0.5 = block identity', ha='left', fontsize=5.8, color=GRAY)
        elif title.startswith('Clock'):
            ax.add_patch(FancyArrowPatch((LX, yc), (xs2[-1][1] + xs2[-1][2], yc), arrowstyle='-|>', mutation_scale=7, lw=0.8, color=INK, zorder=5))
            text(LX + 0.02, yc + 0.1, 'time since session start, scored under each protocol above', ha='left', fontsize=5.8, color=GRAY)
    bottom = y0 - (len(rows) - 1) * row_dy - rh
    assert bottom > 0.02, f'content runs off the bottom: {bottom}'
    save(fig, 'fig1_framework')


# =============================================================== Figure 2: engagement vs difficulty; bracketing rests
def fig2():
    e = pd.read_csv(f'{D}/fig_engagement.csv')
    order = [('COGBCI', 'COG-BCI', 'chest ECG'), ('MAUS', 'MAUS', 'ECG, finger EDA'), ('UNIVERSE', 'UNIVERSE', 'wrist E4'),
             ('MPHRC', 'MultiPhysio-HRC', 'day-1 Stroop'), ('SENSECOBOT', 'SenseCobot', 'ECG, finger EDA')]
    fig = plt.figure(figsize=(W, 2.45))
    ax = fig.add_axes([0.075, 0.25, 0.56, 0.68])
    for i, (ds, lab, sens) in enumerate(order):
        for j, (c, st, nm) in enumerate((('easy_vs_rest', S1, 'easy task vs. rest (engagement)'), ('hard_vs_easy', S2, 'hard vs. easy (difficulty)'))):
            v = e[(e.dataset == ds) & (e.contrast == c)].groupby('pid').auc.mean().values
            if not len(v):
                continue
            x = i + (j - 0.5) * 0.32
            ax.scatter(x + RNG.uniform(-0.06, 0.06, len(v)), v, s=3, color='0.65', lw=0, zorder=2)
            m, lo, hi = boot(v)
            point(ax, x, m, lo, hi, st, label=nm if i == 0 else None)
        n = e[e.dataset == ds].pid.nunique()
        ax.text(i, -0.17, f'{lab}\n{sens}\nn = {n}', ha='center', va='top', fontsize=6.0, transform=ax.get_xaxis_transform())
    chance(ax)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([]); ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_ylim(-0.03, 1.03); ax.set_ylabel('Within-person AUROC\n(leave-one-subject-out model)')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.2, columnspacing=1.2, borderaxespad=0.1)
    letter(None, 0.005, 0.99, 'a', fig=fig)
    # panel b: bracketing rests
    b = pd.read_csv(f'{R3}/bracket.csv'); b = b[b.rep == 'physiology']
    ax = fig.add_axes([0.73, 0.25, 0.26, 0.68])
    items = [('task vs start rest', S1, 'task vs. start-of-session rest'), ('task vs end rest', SC, 'task vs. end-of-session rest'),
             ('end rest vs start rest', S3, 'end rest vs. start rest (drift)')]
    for i, ds in enumerate(('COGBCI', 'MPHRC day 2')):
        for j, (c, st, nm) in enumerate(items):
            q = b[(b.dataset == ds) & (b.contrast == c)].iloc[0]
            lo, hi = [float(x) for x in q.ci.strip('[]').split(',')]
            point(ax, i + (j - 1) * 0.22, q.auc, lo, hi, st, label=nm if i == 0 else None)
    chance(ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['COG-BCI\nn = 29', 'MultiPhysio-HRC\nday 2, n = 42'], fontsize=6.0)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.03, 1.03); ax.set_ylabel('Within-person AUROC')
    ax.legend(loc='lower left', fontsize=5.8, handletextpad=0.2, borderaxespad=0.1, bbox_to_anchor=(0.04, 0.0))
    letter(None, 0.655, 0.99, 'b', fig=fig)
    save(fig, 'fig2_engagement')


# =============================================================== Figure 3: difficulty vs drift in bpm
def fig3():
    t = pd.read_csv(f'{R3}/bpm.csv')
    w = pd.read_csv(f'{R}/confirm/W_confirm.csv')
    order = [('COGBCI', 'COG-BCI', 'n = 29; 8 task blocks (~60 min)'), ('MAUS', 'MAUS', 'n = 22; 6 blocks (30 min)'),
             ('UNIVERSE', 'UNIVERSE', 'n = 24; 8 task blocks (~86 min)'), ('MPHRC', 'MultiPhysio-HRC', 'n = 50; day-1 Stroop, no drift term')]
    fig = plt.figure(figsize=(W, 2.15)); ax = fig.add_axes([0.25, 0.21, 0.72, 0.66])
    for i, (ds, lab, note) in enumerate(order):
        y = len(order) - 1 - i
        for kind, st, dy, nm in (('difficulty', S2, 0.12, 'difficulty: hardest vs. easiest level'), ('drift over one session', S3, -0.12, 'drift: linear session trend of task blocks, extrapolated')):
            q = t[(t.dataset == ds) & (t.kind == kind)]
            if len(q):
                q = q.iloc[0]; point(ax, q.bpm, y + dy, q.lo, q.hi, st, label=nm if i == 0 else None, horizontal=True)
        ax.text(-0.02, y + 0.08, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=6.8, fontweight='bold')
        ax.text(-0.02, y - 0.17, note, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=5.8, color=GRAY)
    chance(ax, 0, horizontal=False)
    ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_ylim(-0.55, len(order) - 0.45)
    ax.set_xlabel('Heart-rate change (beats per minute), mixed model on block means'); ax.set_xlim(-7.5, 5.5)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=6.0, handletextpad=0.2, borderaxespad=0.1, columnspacing=1.5)
    save(fig, 'fig3_drift_vs_difficulty')


# =============================================================== Figure 4: block identity
def fig4():
    n = pd.read_csv(f'{D}/fig_null.csv')
    groups = [('cardiac', dict(S1, facecolor='white'), 'cardiac'), ('eda', S4, 'EDA'), ('temp', dict(S3, facecolor='0.85'), 'skin temperature')]
    dss = [('COG-BCI', 'COG-BCI\nchest'), ('MultiPhysio-HRC', 'MultiPhysio-\nHRC'), ('UNIVERSE', 'UNIVERSE\nwrist'),
           ('WESAD chest', 'WESAD\nchest'), ('WESAD wrist', 'WESAD\nwrist'), ('SenseCobot chest/finger', 'SenseCobot\nchest +\nfinger'),
           ('SenseCobot wrist', 'SenseCobot\nwrist')]
    fig = plt.figure(figsize=(W, 4.1))
    ax = fig.add_axes([0.095, 0.60, 0.50, 0.36])
    vals = []
    for i, (ds, _) in enumerate(dss):
        for j, (g, st, nm) in enumerate(groups):
            v = n[(n.dataset == ds) & (n.group == g)].auc.values
            if len(v) < 5:
                continue
            m, lo, hi = boot(v); point(ax, i + (j - 1) * 0.22, m, lo, hi, st, label=nm if ds == 'UNIVERSE' else None)
            vals.append(dict(dataset=ds, group=g, mean=m, lo=lo, hi=hi, n=len(v)))
    pd.DataFrame(vals).to_csv(f'{D}/fig4a_values.csv', index=False)
    chance(ax); ax.set_ylim(0.3, 1.0); ax.set_xlim(-0.5, len(dss) - 0.5)
    ax.set_xticks(range(len(dss))); ax.set_xticklabels([s for _, s in dss], fontsize=5.6)
    ax.set_ylabel('AUROC, no load difference\n(split-half null)')
    ax.legend(loc='upper left', fontsize=6.0, handletextpad=0.2, borderaxespad=0.1, ncol=3, columnspacing=0.8)
    letter(None, 0.005, 0.99, 'a', fig=fig)
    # b: null vs gap, labelled directly
    s = pd.read_csv(f'{R3}/gapnull_summary.csv', dtype={'gap': str}); s = s[s.k == 6]
    gx = {'orig': 0, '0': 1, '30': 2, '60': 3, '120': 4}
    ax = fig.add_axes([0.69, 0.60, 0.165, 0.36])
    lines = (('WESAD_CHEST', 'temp', dict(S3, facecolor='0.4'), 'WESAD chest, temp.'), ('UNIVERSE', 'temp', dict(S3, facecolor='0.85'), 'UNIVERSE, temp.'),
             ('UNIVERSE', 'eda', S4, 'UNIVERSE, EDA'), ('UNIVERSE', 'cardiac', dict(S1, facecolor='white'), 'UNIVERSE, cardiac'),
             ('COGBCI', 'cardiac', dict(S1, facecolor='black'), 'COG-BCI, cardiac'))
    for ds, grp, st, lab in lines:
        q = s[(s.dataset == ds) & (s.group == grp) & s.gap.isin(gx) & (s['count'] >= 8)].copy(); q['x'] = q.gap.map(gx); q = q.sort_values('x')
        ax.plot(q.x, q['mean'], color=INK, lw=0.6, zorder=2)
        ax.scatter(q.x, q['mean'], zorder=3, label=lab, **dict(st, s=13))
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=5.6, handletextpad=0.1, borderaxespad=0.2, labelspacing=0.5)
    ax.set_ylabel('AUROC, split-half null', fontsize=6.5)
    chance(ax); ax.set_ylim(0.3, 1.0); ax.set_xlim(-0.3, 4.2)
    ax.set_xticks(list(gx.values())); ax.set_xticklabels(['over-\nlap', '0', '0.5', '1', '2'], fontsize=6)
    ax.set_xlabel('Training-test gap (min)', fontsize=6.5)
    letter(None, 0.615, 0.99, 'b', fig=fig)
    # c: protocols vs a time-only model (horizontal)
    gr = pd.read_csv(f'{R3}/gapreal_summary.csv', dtype={'gap': str}); cb = pd.read_csv(f'{R3}/crossblk_summary.csv')
    cs = pd.read_csv(f'{R}/confirm/clock_baseline.csv')
    cats = []
    for ds, lab in (('UNIVERSE', 'UNIVERSE'), ('COGBCI', 'COG-BCI')):
        g = gr[gr.dataset == ds]
        get = lambda prot, gap, rep: g[(g.protocol == prot) & (g.gap == gap) & (g.rep == rep)]['mean'].iloc[0]
        cats.append((f'{lab} · random windows', get('random windows', 'none', 'physiology'), get('random windows', 'none', 'clock')))
        cats.append((f'{lab} · same-session personal (1-min gap)', get('same-session personal', '60', 'physiology'), get('same-session personal', '60', 'clock')))
        c = cs[(cs.dataset == ds) & (cs.protocol == 'cross-session personal')]
        cats.append((f'{lab} · cross-session personal', c[c.rep == 'physiology'].groupby('pid').auc.mean().mean(), c[c.rep == 'clock'].groupby('pid').auc.mean().mean()))
    m = cb[cb.dataset == 'MAUS_ECG']
    cats.append(('MAUS · same-session personal', m[(m.protocol.str.startswith('same')) & (m.rep == 'physiology')]['mean'].iloc[0],
                 m[(m.protocol.str.startswith('same')) & (m.rep == 'clock')]['mean'].iloc[0]))
    cats.append(('MAUS · cross-block (same day)', m[(m.protocol == 'train blocks 1-3, test 4-6') & (m.rep == 'physiology')]['mean'].iloc[0],
                 m[(m.protocol == 'train blocks 1-3, test 4-6') & (m.rep == 'clock')]['mean'].iloc[0]))
    ax = fig.add_axes([0.32, 0.10, 0.45, 0.32])
    for i, (lbl, p, c) in enumerate(cats):
        y = len(cats) - 1 - i
        ax.plot([p, c], [y, y], color='0.7', lw=0.6, zorder=2)
        ax.scatter([p], [y], zorder=4, label='physiological features' if i == 0 else None, **S2)
        ax.scatter([c], [y], zorder=6, label='time only (clock)' if i == 0 else None, marker='x', s=18, color='black', linewidth=1.0)
        ax.text(-0.03, y, lbl, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=6.1)
    chance(ax, horizontal=False); ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.6, len(cats) - 0.4)
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xlabel('Within-person AUROC, hard vs. easy (personal models; MAUS: 3- vs. 0-back)', fontsize=6.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=6.0, handletextpad=0.2)
    letter(None, 0.005, 0.455, 'c', fig=fig)
    ax0 = fig.add_axes([0, 0, 1, 1], facecolor='none'); ax0.axis('off')
    ax0.add_line(Line2D([0.01, 0.99], [0.47, 0.47], color='0.6', lw=0.4))
    pd.DataFrame(cats, columns=['protocol', 'physiology', 'clock']).to_csv(f'{D}/fig4c_values.csv', index=False)
    save(fig, 'fig4_block_identity')


# =============================================================== Figure 5: sensors
def fig5():
    u = pd.read_csv(f'{R3}/usable.csv'); u = u[u.thr == 0.8]
    rows = [('MAUS torso ECG', 'rest', 'task', 'MAUS · torso ECG'), ('MAUS wrist PPG', 'rest', 'task', 'MAUS · wrist PPG (idle hand)'),
            ('CogWear wrist E4 (BVP)', 'rest', 'task', 'CogWear · wrist E4'), ('CogWear Galaxy Watch4 (PPG)', 'rest', 'task', 'CogWear · Galaxy Watch4'),
            ('WESAD chest ECG', 'rest', 'stress (TSST)', 'WESAD · chest ECG'), ('WESAD wrist E4 (BVP)', 'rest', 'stress (TSST)', 'WESAD · wrist E4'),
            ('SenseCobot chest ECG', 'rest', 'task', 'SenseCobot · chest ECG (9 pass QC)'),
            ('SenseCobot wrist E4 (vendor IBI)', 'rest', 'task', 'SenseCobot · E4 beat-interval export')]
    fig = plt.figure(figsize=(W, 2.5))
    ax = fig.add_axes([0.30, 0.20, 0.30, 0.73])
    for i, (dev, cr, ct, lab) in enumerate(rows):
        y = len(rows) - 1 - i
        a = u[(u.device == dev) & (u.cond == cr)]; b = u[(u.device == dev) & (u.cond == ct)]
        va, vb = 100 * a.pooled.iloc[0], 100 * b.pooled.iloc[0]
        ax.plot([va, vb], [y, y], color='0.55', lw=0.8, zorder=2)
        ax.scatter([va], [y], zorder=4, label='seated rest' if i == 0 else None, marker='o', s=42, facecolor='white', edgecolor='black', linewidth=0.8)
        ax.scatter([vb], [y], zorder=5, label='task (WESAD: stress task)' if i == 0 else None, marker='o', s=12, facecolor='black', edgecolor='black', linewidth=0.4)
        ax.text(-0.02, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=6.2)
        if dev.startswith('SenseCobot wrist'):
            ax.text(va + 4, y, f'{100 * b.share_no_data.iloc[0]:.0f}% of task windows: no export file', fontsize=5.6, color=GRAY, va='center')
    ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlim(-2, 102); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel('Windows with usable beats (%, ≥ 80% valid intervals)', fontsize=6.5)
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.40, 1.0), ncol=2, fontsize=6.0, handletextpad=0.2, frameon=False)
    letter(None, 0.005, 0.99, 'a', fig=fig)
    g = pd.read_csv(f'{R3}/agree.csv')
    pairs = [('MAUS wrist PPG vs torso ECG (all estimate windows)', 'rest', 'task', 'MAUS wrist PPG vs. torso ECG'),
             ('CogWear Galaxy Watch4 vs wrist E4 (all estimate windows)', 'rest', 'task', 'CogWear Galaxy Watch4 vs. wrist E4'),
             ('WESAD wrist E4 vs chest ECG (all estimate windows)', 'rest', 'stress (TSST)', 'WESAD wrist E4 vs. chest ECG'),
             ('SenseCobot E4 HR column vs chest ECG (all estimate windows)', 'rest', 'task', 'SenseCobot E4 heart rate vs. ECG'),
             ('SenseCobot Bangle.js HR (conf >= 80) vs chest ECG (all estimate windows)', 'rest', 'task', 'SenseCobot Bangle.js vs. ECG')]
    ax = fig.add_axes([0.80, 0.20, 0.19, 0.73])
    for i, (p, cr, ct, lab) in enumerate(pairs):
        y = len(pairs) - 1 - i
        a = g[(g.pair == p) & (g.cond == cr)].iloc[0]; b = g[(g.pair == p) & (g.cond == ct)].iloc[0]
        ax.plot([100 * a.within5, 100 * b.within5], [y, y], color='0.55', lw=0.8, zorder=2)
        ax.scatter([100 * a.within5], [y], zorder=4, marker='o', s=42, facecolor='white', edgecolor='black', linewidth=0.8)
        ax.scatter([100 * b.within5], [y], zorder=5, marker='o', s=12, facecolor='black', edgecolor='black', linewidth=0.4)
        ax.text(-0.04, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=6.0)
    ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlim(0, 102); ax.set_ylim(-0.6, len(pairs) - 0.4)
    ax.set_xlabel('Wrist windows within ±5 bpm\nof the reference (%)', fontsize=6.5)
    letter(None, 0.615, 0.99, 'b', fig=fig)
    save(fig, 'fig5_sensors')


# =============================================================== Figure 6: audit
def fig6():
    s = json.load(open('/path/to/project/lit/audit/M6b_summary.json'))
    bars = [('Within-person split', 17, [('random windows', s['E2']['a_random_windows'], '0.25', 'white'), ('same session, by time or block', s['E2']['b_within_session_temporal'], '0.6', 'black'),
                                         ('different day', s['n_cross_session'], '0.85', 'black'), ('unclear', s['E2']['d_unclear'], 'white', 'black')]),
            ('Condition order', 40, [('fixed', s['E4']['fixed'], '0.25', 'white'), ('randomised or counterbalanced', s['E4']['randomized_or_counterbalanced'], '0.6', 'black'),
                                     ('not reported', s['E4']['not_reported'], 'white', 'black')]),
            ('Manipulation check', 40, [('reported', s['E5']['reported'], '0.6', 'black'), ('label is a self-report', s['E5']['label_is_self_report'], '0.85', 'black'),
                                        ('not reported', s['E5']['not_reported'], 'white', 'black')])]
    fig = plt.figure(figsize=(W, 1.5)); ax = fig.add_axes([0.25, 0.03, 0.73, 0.94])
    for i, (lab, tot, segs) in enumerate(bars):
        y = len(bars) - 1 - i; x = 0
        for name, v, fc, tc in segs:
            if v == 0:
                continue
            w = 100 * v / tot
            ax.barh(y, w, left=x, height=0.42, color=fc, edgecolor='black', linewidth=0.5)
            pct = int(100 * v / tot + 0.5)   # round half up, as in the text
            ax.text(x + w / 2, y, f'{v} ({pct}%)', ha='center', va='center', fontsize=6.0, color=tc)
            ax.text(x + w / 2, y - 0.27, name, ha='center', va='top', fontsize=5.7, color=GRAY)
            x += w
        ax.text(-1.5, y, f'{lab}\n({tot} studies' + ('; none across days)' if tot == 17 else ')'), ha='right', va='center', fontsize=6.6)
    ax.set_xlim(0, 100); ax.set_ylim(-0.75, len(bars) - 0.55); ax.axis('off')
    save(fig, 'fig6_audit')


# =============================================================== R5 figures (7 datasets, focused on block identity)
def _ci_str(c):
    lo, hi = [float(x) for x in str(c).strip('[]').split(',')]
    return lo, hi


def fig2_r5():
    """Effect sizes in bpm (difficulty vs session drift); same content as the R4 Figure 3."""
    fig3()
    import shutil
    for ext in ('pdf', 'png'):
        shutil.copy(f'{FIG}/fig3_drift_vs_difficulty.{ext}', f'{FIG}/fig2_effects.{ext}')


def fig3_r5():
    # a: personal split-half null and the part shared across people (direction learned from other people), same test windows
    n = pd.read_csv(f'{R3}/r6_drift_long.csv')
    n['dataset'] = n.dataset.map({'COGBCI': 'COG-BCI', 'MPHRC': 'MultiPhysio-HRC', 'UNIVERSE': 'UNIVERSE', 'WESAD_CHEST': 'WESAD chest',
                                  'WESAD_WRIST': 'WESAD wrist', 'MAUS_WRIST': 'MAUS wrist'})
    n = n.groupby(['dataset', 'group', 'pid'])[['personal', 'population']].mean().reset_index()
    groups = [('cardiac', dict(S1, facecolor='white'), 'cardiac'), ('eda', S4, 'EDA'), ('temp', dict(S3, facecolor='0.85'), 'skin temperature')]
    dss = [('COG-BCI', 'COG-BCI\nchest'), ('MultiPhysio-HRC', 'MultiPhysio-\nHRC'), ('UNIVERSE', 'UNIVERSE\nwrist'),
           ('WESAD chest', 'WESAD\nchest'), ('WESAD wrist', 'WESAD\nwrist'), ('MAUS wrist', 'MAUS\nwrist PPG')]
    fig = plt.figure(figsize=(W, 4.1))
    ax = fig.add_axes([0.095, 0.60, 0.50, 0.36]); vals = []
    for i, (ds, _) in enumerate(dss):
        for j, (g, st, nm) in enumerate(groups):
            q = n[(n.dataset == ds) & (n.group == g)]
            v = q.personal.values
            if np.isfinite(v).sum() < 5:
                continue
            x = i + (j - 1) * 0.25
            m, lo, hi = boot(v); point(ax, x - 0.045, m, lo, hi, st, label=nm if ds == 'UNIVERSE' else None)
            pm, plo, phi = boot(q.population.values)
            ax.plot([x + 0.065, x + 0.065], [plo, phi], color='0.6', lw=0.7, zorder=2)
            ax.plot([x - 0.045, x + 0.025], [m, pm], color='0.55', lw=0.5, zorder=2)
            ax.plot([x + 0.025, x + 0.105], [pm, pm], color=INK, lw=1.5, solid_capstyle='butt', zorder=5,
                    label='population null (direction learned from others)' if (ds, g) == ('COG-BCI', 'cardiac') else None)
            vals.append(dict(dataset=ds, group=g, mean=m, lo=lo, hi=hi, population=pm, pop_lo=plo, pop_hi=phi, n=int(np.isfinite(v).sum())))
    pd.DataFrame(vals).to_csv(f'{D}/fig3a_values.csv', index=False)
    chance(ax); ax.set_ylim(0.3, 1.0); ax.set_xlim(-0.5, len(dss) - 0.5)
    ax.set_xticks(range(len(dss))); ax.set_xticklabels([s_ for _, s_ in dss], fontsize=5.9)
    ax.set_ylabel('AUROC, no load difference\n(split-half null)')
    h, l = ax.get_legend_handles_labels(); order = [l.index(x) for x in ('cardiac', 'EDA', 'skin temperature', 'population null (direction learned from others)')]
    ax.legend([h[o] for o in order], [l[o] for o in order], loc='lower right', fontsize=5.8, handletextpad=0.3, borderaxespad=0.2,
              ncol=2, columnspacing=0.8, handlelength=1.2, labelspacing=0.3)
    letter(None, 0.005, 0.99, 'a', fig=fig)
    s_ = pd.read_csv(f'{R3}/gapnull_summary.csv', dtype={'gap': str}); s_ = s_[s_.k == 6]
    gx = {'orig': 0, '0': 1, '30': 2, '60': 3, '120': 4}
    ax = fig.add_axes([0.69, 0.60, 0.165, 0.36])
    lines = (('WESAD_CHEST', 'temp', dict(S3, facecolor='0.4'), 'WESAD chest, temp.'), ('UNIVERSE', 'temp', dict(S3, facecolor='0.85'), 'UNIVERSE, temp.'),
             ('UNIVERSE', 'eda', S4, 'UNIVERSE, EDA'), ('UNIVERSE', 'cardiac', dict(S1, facecolor='white'), 'UNIVERSE, cardiac'),
             ('COGBCI', 'cardiac', dict(S1, facecolor='black'), 'COG-BCI, cardiac'))
    for ds, grp, st, lab in lines:
        q = s_[(s_.dataset == ds) & (s_.group == grp) & s_.gap.isin(gx) & (s_['count'] >= 8)].copy(); q['x'] = q.gap.map(gx); q = q.sort_values('x')
        ax.plot(q.x, q['mean'], color=INK, lw=0.6, zorder=2); ax.scatter(q.x, q['mean'], zorder=3, label=lab, **dict(st, s=13))
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=5.6, handletextpad=0.1, borderaxespad=0.2, labelspacing=0.5)
    chance(ax); ax.set_ylim(0.3, 1.0); ax.set_xlim(-0.3, 4.2); ax.set_ylabel('AUROC, split-half null', fontsize=6.5)
    ax.set_xticks(list(gx.values())); ax.set_xticklabels(['over-\nlap', '0', '0.5', '1', '2'], fontsize=6)
    ax.set_xlabel('Training-test gap (min)', fontsize=6.5)
    letter(None, 0.615, 0.99, 'b', fig=fig)
    # c: same-condition, different blocks
    ma = pd.read_csv(f'{R}/prereg_r2/MAUS_prereg.csv'); sl = pd.read_csv(f'{R3}/r5_samelevel_ds007554_summary.csv', dtype={'gap': str, 'apart': str})
    rows = [('MAUS · finger EDA (two 0-, 2- or 3-back blocks)', ma.loc[6, 'estimate'], *_ci_str(ma.loc[6, 'ci']), S4),
            ('MAUS · wrist PPG cardiac', ma.loc[7, 'estimate'], *_ci_str(ma.loc[7, 'ci']), dict(S1, facecolor='white')),
            ('MAUS · torso ECG cardiac', ma.loc[5, 'estimate'], *_ci_str(ma.loc[5, 'ci']), dict(S1, facecolor='white'))]
    for ap, lab in (('1', 'adjacent repetitions'), ('2', 'repetitions two apart')):
        for gp, st, gl in (('orig', dict(S1, facecolor='white'), ''), ('60', dict(S1, facecolor='0.6'), ', 1-min gap')):
            q = sl[(sl.apart == ap) & (sl.gap == gp)].iloc[0]
            rows.append((f'ds007554 · chest ECG, {lab}{gl}', q.auc, *_ci_str(q.ci), st))
    ax = fig.add_axes([0.40, 0.08, 0.40, 0.32])
    for i, (lab, m, lo, hi, st) in enumerate(rows):
        y = len(rows) - 1 - i; point(ax, m, y, lo, hi, dict(st, s=18), horizontal=True)
        ax.text(-0.03, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=6.0)
    chance(ax, horizontal=False); ax.set_xlim(0.3, 1.0); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xlabel('AUROC separating two blocks of the same condition (personal model)', fontsize=6.5)
    letter(None, 0.005, 0.455, 'c', fig=fig)
    ax0 = fig.add_axes([0, 0, 1, 1], facecolor='none'); ax0.axis('off')
    ax0.add_line(Line2D([0.01, 0.99], [0.47, 0.47], color='0.6', lw=0.4))
    pd.DataFrame([r[:4] for r in rows], columns=['row', 'auc', 'lo', 'hi']).to_csv(f'{D}/fig3c_values.csv', index=False)
    save(fig, 'fig3_block_identity')


def fig4_r5():
    """Hard vs easy under each protocol. Physiology from the R7 run (k = 6, C = 1; same training windows for same-session
    and cross-session/cross-block), clock from the clock-baseline run."""
    pr = pd.read_csv(f'{R3}/r7_protocols_long.csv'); pr = pr[(pr.C == 1.0) & (pr.k.isna() | (pr.k == 6))]
    gr = pd.read_csv(f'{R3}/gapreal_summary.csv', dtype={'gap': str})
    cs = pd.read_csv(f'{R}/confirm/clock_baseline.csv')
    mi = pd.read_csv(f'{R3}/r7_maus_inflation_long.csv'); lb = pd.read_csv(f'{R3}/r7_maus_lobpo_long.csv')
    cw = pd.read_csv(f'{R3}/r7_cogwear_long.csv')
    rows = []
    def bt(ds, prot, rep):
        v = cs[(cs.dataset == ds) & (cs.protocol == prot) & (cs.rep == rep)].groupby('pid').auc.mean().values
        return boot(v)
    for ds, lab, feats in (('UNIVERSE', 'UNIVERSE', 'all features'), ('UNIVERSE', 'UNIVERSE, cardiac only', 'cardiac only'), ('COGBCI', 'COG-BCI', 'all features')):
        g = pr[(pr.dataset == ds) & (pr.features == feats)]
        clk = {'random windows': gr[(gr.dataset == ds) & (gr.protocol == 'random windows') & (gr.rep == 'clock')]['mean'].iloc[0],
               'same-session personal (1-min gap)': gr[(gr.dataset == ds) & (gr.protocol == 'same-session personal') & (gr.gap == '60') & (gr.rep == 'clock')]['mean'].iloc[0],
               'population (LOSO)': bt(ds, 'population (LOSO)', 'clock')[0], 'cross-session personal': bt(ds, 'cross-session personal', 'clock')[0]}
        for prot in ('random windows', 'same-session personal (1-min gap)', 'population (LOSO)', 'cross-session personal'):
            v = g[g.protocol == prot].groupby('pid').auc.mean().values; m, lo, hi = boot(v)
            rows.append((lab, prot, m, lo, hi, clk[prot], len(v)))
    for col, name, clk in (('same_session', 'same-session personal', 1.0), ('cross_block', 'cross-block (later blocks; same training windows)', 0.0)):
        v = mi[(mi.dataset == 'MAUS_ECG') & (mi.rep == 'physiology')][col].values; m, lo, hi = boot(v)
        rows.append(('MAUS', name, m, lo, hi, clk, len(v)))
    v = lb[(lb.dataset == 'MAUS_ECG') & (lb.rep == 'physiology')].groupby('pid').auc.mean().values; m, lo, hi = boot(v)
    rows.append(('MAUS', 'leave-one-block-pair-out', m, lo, hi, 0.0, len(v)))
    a = cw[cw.features == 'all features'].dropna(subset=['same-session personal', 'cross-session personal'])
    for col in ('same-session personal', 'cross-session personal'):
        m, lo, hi = boot(a[col].values); rows.append(('CogWear', col, m, lo, hi, np.nan, len(a)))
    # group headers: what there was to detect in each dataset (Section 4.1)
    HEAD = {'UNIVERSE': 'UNIVERSE: no heart-rate difficulty effect (−0.2 bpm)', 'COG-BCI': 'COG-BCI: heart-rate difficulty effect 2.1 bpm',
            'MAUS': 'MAUS: heart-rate difficulty effect 3.4 bpm', 'CogWear': 'CogWear: Stroop vs. baseline (task vs. rest)'}
    fig = plt.figure(figsize=(W, 4.45)); ax = fig.add_axes([0.38, 0.10, 0.40, 0.89])
    y = 0; ys = []; first = True
    prev = None
    for ds, name, p, lo, hi, c, nn in rows:
        if prev is not None and ds != prev:
            y -= 0.6 if ds not in HEAD else 1.15
        if ds != prev and ds in HEAD:
            ax.text(-0.03, y + 0.72, HEAD[ds], transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=6.1, fontweight='bold', color=INK)
        prev = ds
        if np.isfinite(lo):
            ax.plot([lo, hi], [y, y], color=INK, lw=0.9, zorder=3)
        ax.scatter([p], [y], zorder=4, label='physiological features' if first else None, **S2)
        if np.isfinite(c):
            ax.scatter([c], [y], zorder=6, label='time only (clock)' if first else None, marker='x', s=18, color='black', linewidth=1.0)
        rl = name if ds in HEAD else ('cardiac features only · ' + name if 'cardiac only' in ds else f'{ds} · {name}')
        ax.text(-0.03, y, rl, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=6.0)
        ax.text(1.015, y, f'n = {nn}', transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=5.6, color=GRAY)
        ys.append(y); y -= 1; first = False
    chance(ax, horizontal=False); ax.set_xlim(-0.03, 1.03); ax.set_ylim(min(ys) - 0.6, 1.15)
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xlabel('Within-person AUROC, hard vs. easy (MAUS: 3- vs. 0-back; CogWear: Stroop vs. baseline)', fontsize=6.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1.10, 1.0), fontsize=6.0, handletextpad=0.2)
    pd.DataFrame(rows, columns=['dataset', 'protocol', 'physiology', 'lo', 'hi', 'clock', 'n']).to_csv(f'{D}/fig4_values.csv', index=False)
    save(fig, 'fig4_protocols')


def fig5_r5():
    u = pd.read_csv(f'{R3}/usable.csv'); u = u[u.thr == 0.8]
    rows = [('COG-BCI chest ECG', 'rest', 'task', 'COG-BCI · chest ECG'), ('MAUS torso ECG', 'rest', 'task', 'MAUS · torso ECG'),
            ('WESAD chest ECG', 'rest', 'stress (TSST)', 'WESAD · chest ECG'), ('MAUS wrist PPG', 'rest', 'task', 'MAUS · wrist PPG (idle hand)'),
            ('CogWear wrist E4 (BVP)', 'rest', 'task', 'CogWear · wrist E4'), ('UNIVERSE wrist E4 (BVP)', 'rest', 'task', 'UNIVERSE · wrist E4'),
            ('WESAD wrist E4 (BVP)', 'rest', 'stress (TSST)', 'WESAD · wrist E4')]
    fig = plt.figure(figsize=(W, 2.3)); ax = fig.add_axes([0.28, 0.21, 0.32, 0.70])
    for i, (dev, cr, ct, lab) in enumerate(rows):
        y = len(rows) - 1 - i
        a = u[(u.device == dev) & (u.cond == cr)]; b = u[(u.device == dev) & (u.cond == ct)]
        va, vb = 100 * a.pooled.iloc[0], 100 * b.pooled.iloc[0]
        ax.plot([va, vb], [y, y], color='0.55', lw=0.8, zorder=2)
        ax.scatter([va], [y], zorder=4, label='seated rest' if i == 0 else None, marker='o', s=42, facecolor='white', edgecolor='black', linewidth=0.8)
        ax.scatter([vb], [y], zorder=5, label='task (WESAD: stress task)' if i == 0 else None, marker='o', s=12, facecolor='black', edgecolor='black', linewidth=0.4)
        ax.text(-0.02, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=6.2)
    ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlim(-2, 102); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel('Windows with usable beats (%, ≥ 80% valid intervals)', fontsize=6.5)
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.40, 1.0), ncol=2, fontsize=6.0, handletextpad=0.2, frameon=False)
    letter(None, 0.005, 0.99, 'a', fig=fig)
    g = pd.read_csv(f'{R3}/agree.csv')
    pairs = [('MAUS fingertip PPG vs torso ECG (all estimate windows)', 'rest', 'task', 'MAUS fingertip PPG vs. torso ECG'),
             ('MAUS wrist PPG vs torso ECG (all estimate windows)', 'rest', 'task', 'MAUS wrist PPG vs. torso ECG'),
             ('WESAD wrist E4 vs chest ECG (all estimate windows)', 'rest', 'stress (TSST)', 'WESAD wrist E4 vs. chest ECG')]
    ax = fig.add_axes([0.82, 0.21, 0.16, 0.70])
    for i, (p, cr, ct, lab) in enumerate(pairs):
        y = len(pairs) - 1 - i
        a = g[(g.pair == p) & (g.cond == cr)].iloc[0]; b = g[(g.pair == p) & (g.cond == ct)].iloc[0]
        ax.plot([100 * a.within5, 100 * b.within5], [y, y], color='0.55', lw=0.8, zorder=2)
        ax.scatter([100 * a.within5], [y], zorder=4, marker='o', s=42, facecolor='white', edgecolor='black', linewidth=0.8)
        ax.scatter([100 * b.within5], [y], zorder=5, marker='o', s=12, facecolor='black', edgecolor='black', linewidth=0.4)
        ax.text(-0.04, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=6.0)
    ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlim(0, 102); ax.set_ylim(-0.6, len(pairs) - 0.4)
    ax.set_xlabel('Windows within ±5 bpm\nof the ECG (%)', fontsize=6.5)
    letter(None, 0.615, 0.99, 'b', fig=fig)
    save(fig, 'fig5_signal_quality')


if __name__ == '__main__':
    os.environ.setdefault('DATA_ROOT', '/path/to/project/data/raw')
    which = sys.argv[1:] or ['1', '2_r5', '3_r5', '4_r5', '5_r5']
    for k in which:
        globals()[f'fig{k}'](); print('done fig', k, flush=True)
    print('shrunk:', SHRUNK)
