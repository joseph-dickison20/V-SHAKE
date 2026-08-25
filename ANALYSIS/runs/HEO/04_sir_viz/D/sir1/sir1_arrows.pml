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
cmd.load('/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/HEO/04_sir_viz/D/sir1/sir1_arrows.xyz', 'mol')
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
cmd.load('/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/TRAJS/xDENPLT/OUTPUT/HEO_D.cube', 'dens')
cmd.hide('everything', 'dens')
cmd.isomesh('density', 'dens', 0.04)
cmd.set_color('denscol', [128, 229, 255])
cmd.color('denscol', 'density')
cmd.set('state', 1)
cmd.reset()
cmd.turn('x', 0.0)
cmd.turn('y', 180.0)
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
bases = [[1.1949655599999995, 0.8506201600000001, -1.000000000000001e-08], [-1.21675875, 0.8140536300000001, 2.9999999999999964e-08], [1.1888516199999994, -0.43503535, -9.99999999999999e-09], [-1.21770142, -0.4716057, 2.999999999999999e-08], [0.005419849999999893, -1.17931957, 1.0000000000000017e-08], [0.032802139999999924, -2.26531147, 1.0000000000000037e-08], [2.1685176099999994, -0.93390577, -1.999999999999996e-08], [-2.5517851999999994, -1.1608726, 8.999999999999999e-08], [-3.1230641599999998, -0.84649899, -0.884756], [-2.4477196599999997, -2.25141892, -1.4e-07], [-3.1230638799999997, -0.84649936, 0.8847565]]
tips = [[-0.8355153490894192, 1.2864379770023047, -0.03773563558319797], [0.6148863423756956, 1.118750634454074, 0.26736804238614], [0.9366195738903788, -0.8706002653556295, -0.23594935552656], [-0.7950887761569417, -0.5034513803453156, -0.263195102869527], [-0.16774601959993765, -1.5648656261403906, -0.011093236356771854], [0.10283441287806494, -2.2049180462615685, 0.13979435260261894], [2.4328704324380874, -0.9513237670130769, 0.14209116627263652], [-2.27797827407153, -0.8194372175063394, -0.021745016925636372], [-3.256143833071716, -1.005798497059418, -0.9175099190745869], [-2.507041134107563, -2.2937292527829136, -0.06743599977388216], [-3.3372336654851193, -0.9168584989917249, 1.0054111948487643]]
r, g, b = _rgb('#1f77b4')
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
png = '/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/HEO/04_sir_viz/D/sir1/sir1_arrows.png'
cmd.png(png, width=W, height=H, dpi=DPI, ray=1)
cmd.quit()
python end
