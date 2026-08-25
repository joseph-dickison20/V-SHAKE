python
from pymol import cmd
from pymol.cgo import CYLINDER, CONE
import os
cmd.reinitialize()
cmd.set('auto_zoom', 0)
cmd.bg_color('white')
cmd.set('ray_opaque_background', 1)
cmd.set('ambient', 0.2)
cmd.set('ray_shadows', 0)
cmd.set('spec_reflect', 0.6)
cmd.set('spec_power', 600)
cmd.set('spec_count', 3)
cmd.set('shininess', 70)
cmd.set('reflect', 0.5)
cmd.set('ray_trace_mode', 1)
cmd.set('ray_texture', 2)
cmd.set('antialias', 3)
cmd.set('fog', 0)
cmd.load('/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/CBO/04_sir_viz/H/sir2/sir2_arrows.xyz', 'mol')
cmd.hide('lines', 'mol')
cmd.show('sticks', 'mol')
cmd.set_bond('stick_radius', 0.08, 'mol')
cmd.show('spheres', 'mol')
cmd.set('sphere_scale', 0.2, 'mol')
cmd.color('gray60', '(mol and elem C)')
cmd.set('stick_color', 'gray90', 'mol')
cmd.set('sphere_transparency', 0.2)
cmd.set('stick_transparency', 0.2)
cmd.set('sphere_transparency', 0.2, 'mol')
cmd.set('stick_transparency', 0.2, 'mol')
cmd.load('/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/TRAJS/xDENPLT/OUTPUT/CBO_H.cube', 'dens')
cmd.hide('everything', 'dens')
cmd.isomesh('density', 'dens', 0.04)
cmd.set_color('denscol', [128, 229, 255])
cmd.color('denscol', 'density')
cmd.set('state', 1)
cmd.reset()
cmd.turn('x', 0.0)
cmd.turn('y', 270.0)
cmd.turn('z', 180.0)
cmd.center('mol')
cmd.zoom('center', 3.25)
view = cmd.get_view()
SHAFT_R = 0.03
HEAD_R = 0.08
HEAD_FRAC = 0.3
MIN_LEN = 1e-06
def _rgb(col_spec):
    if col_spec.startswith('0x') or col_spec.startswith('#'):
        _h = col_spec[2:] if col_spec.startswith('0x') else col_spec[1:]
        return tuple(int(_h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return cmd.get_color_tuple(cmd.get_color_index(col_spec))
bases = [[-0.11079062999999978, 0.0037302700000004796, -0.09807844999999937], [0.36732744999999967, -0.009315849999999792, 2.5662143199999994], [-0.46088386000000037, 1.0723528899999997, 2.727775999999999], [-0.6250722099999999, 1.9346357899999995, 1.4451455899999996], [0.3526578500000005, 2.382259229999999, 1.1914667099999998], [-1.32568261, 2.765888779999999, 1.6500889399999994], [-1.13272176, 1.11312606, 0.25129111000000015], [-1.3473591699999994, 1.7615964699999997, -0.6175246499999996], [-2.08688427, 0.6371475000000002, 0.5319281100000001], [-0.6194439999999999, -1.0245988299999989, -0.9746105299999988], [-1.03386446, -1.8776006799999987, -1.6599202599999987], [0.7969716500000007, 0.4368221300000002, -0.5467987499999991], [-1.5074253100000008, 0.7828565899999999, 3.038378689999999], [-0.11113677000000033, 1.7667671099999993, 3.539927259999999]]
tips = [[-0.06863367353404985, 0.338142297659212, -0.2643990959134004], [0.3875081849297424, 0.019211773893295148, 1.731534967942296], [-0.11543232819581933, 1.1549044336377152, 3.3028013870235076], [-0.01849107038901321, 1.6886306308135266, 2.5910090984856806], [-0.08208200065868954, 1.6905849939375557, 1.1372309361238793], [-0.9893065446060341, 1.7255081278926774, 1.5882423184309622], [-0.9721277923065432, 1.5336066716952126, -0.873904024786847], [-1.4629565420093928, 1.5359876472382854, -1.287060446244604], [-2.290451035302447, 0.9122667598129282, 0.8203793758727818], [-0.2500651078548297, -0.934658798926442, -0.5445847638432595], [-1.3097839851756699, -1.557416839119206, -1.0463887737814592], [0.8022716015185911, 0.799357258603679, -0.40649049815599636], [-1.9869993106213801, 0.7562262479402939, 3.3238813118051467], [-0.4877584957944578, 2.0833162549212685, 2.9730322970413168]]
r, g, b = _rgb('#d62728')
arrow = []
for base, tip in zip(bases, tips):
    dx, dy, dz = tip[0] - base[0], tip[1] - base[1], tip[2] - base[2]
    length = (dx * dx + dy * dy + dz * dz) ** 0.5
    if length < MIN_LEN:
        continue
    sx = base[0] + (1.0 - HEAD_FRAC) * dx
    sy = base[1] + (1.0 - HEAD_FRAC) * dy
    sz = base[2] + (1.0 - HEAD_FRAC) * dz
    arrow += [CYLINDER, base[0], base[1], base[2], sx, sy, sz, SHAFT_R, r, g, b, r, g, b]
    arrow += [CONE, sx, sy, sz, tip[0], tip[1], tip[2], HEAD_R, 0.0, r, g, b, r, g, b, 1.0, 1.0]
cmd.load_cgo(arrow, 'arrows0')
cmd.set_view(view)
W, H, DPI = 2000, 1400, 600
png = '/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/CBO/04_sir_viz/H/sir2/sir2_arrows.png'
cmd.png(png, width=W, height=H, dpi=DPI, ray=1)
cmd.quit()
python end
