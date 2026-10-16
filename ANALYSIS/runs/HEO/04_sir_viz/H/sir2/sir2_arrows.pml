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
cmd.load('/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/HEO/04_sir_viz/H/sir2/sir2_arrows.xyz', 'mol')
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
cmd.load('/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/TRAJS/xDENPLT/OUTPUT/HEO_H.cube', 'dens')
cmd.hide('everything', 'dens')
cmd.isomesh('density', 'dens', 0.04)
cmd.set_color('denscol', [128, 229, 255])
cmd.color('denscol', 'density')
cmd.set('state', 1)
cmd.reset()
cmd.turn('x', 0.0)
cmd.turn('y', 0.0)
cmd.turn('z', 0.0)
cmd.center('mol')
cmd.zoom('center', 3.0)
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
bases = [[-1.191031710000001, 0.8506807000000003, 1.0000000000000268e-08], [1.2124106500000003, 0.8152615900000004, -2.9999999999999706e-08], [-1.187968200000001, -0.4350558699999999, 1.0000000000000037e-08], [1.2170405400000002, -0.47040827999999996, -2.999999999999994e-08], [-0.005017680000000357, -1.18021232, -1.0000000000000083e-08], [-0.03174102000000034, -2.2661797000000004, -1.0000000000000275e-08], [-2.1684127500000008, -0.93239249, 1.9999999999999934e-08], [2.5525396600000008, -1.15679343, -9.000000000000004e-08], [3.123092950000001, -0.8411435100000001, 0.88477152], [2.4507176300000006, -2.2475562900000003, 1.3999999999999977e-07], [3.123092680000001, -0.8411438800000001, -0.88477202]]
tips = [[-0.8222371620825033, 1.312438311568076, 0.6133862008249903], [-0.45024473618465954, 0.06556366203672104, -0.6250608582379134], [-0.8260205566222023, -0.07517129376122128, -0.5521837608990087], [1.6199748098056679, -1.0215169605591743, 0.926564969950697], [0.3927180720967268, -1.000430143785549, 0.471391165477618], [-0.05679212614495131, -2.3607633123360956, 0.07303249907759563], [-2.1385836041885486, -0.7669328586089725, -0.39258171109493667], [3.4240834263879982, -1.377795808165662, 0.07248864757053937], [2.3779434854575277, -0.9717315866133074, 0.21558332709161765], [2.547025670324829, -1.8429344527012692, 0.10060522394649175], [3.0268554711501188, -0.6656690370735449, -0.9032261937076929]]
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
png = '/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/HEO/04_sir_viz/H/sir2/sir2_arrows.png'
cmd.png(png, width=W, height=H, dpi=DPI, ray=1)
cmd.quit()
python end
