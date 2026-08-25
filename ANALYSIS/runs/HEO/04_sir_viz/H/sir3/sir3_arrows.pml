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
cmd.load('/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/HEO/04_sir_viz/H/sir3/sir3_arrows.xyz', 'mol')
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
tips = [[-1.3953508071428489, 0.4049251248887141, -0.3229269436660466], [2.6135229239691062, 0.7822360925374956, 0.6316547909731203], [-1.711479509605092, -0.7454398094144279, 0.7577722634157327], [1.0591998908514633, -0.4392449011031184, 0.21051350219639], [-0.4536589401105516, -1.3876570580898848, -1.520364017071623], [-0.03508332083010019, -1.9131455135199567, 0.6358647673463202], [-2.872461482213115, -0.9084588814703501, -0.03639989366203113], [2.445636565828337, -1.0236662508688426, 0.11062675599116704], [3.2551185681792836, -0.6197712557545392, 1.0484893157932618], [2.182301509624665, -1.8594203313492446, 0.09330836706209672], [4.006977351448852, -0.9953006958558472, -1.6085393983783889]]
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
png = '/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/HEO/04_sir_viz/H/sir3/sir3_arrows.png'
cmd.png(png, width=W, height=H, dpi=DPI, ray=1)
cmd.quit()
python end
