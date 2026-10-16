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
cmd.load('/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/HEO/04_sir_viz/D/sir2/sir2_arrows.xyz', 'mol')
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
tips = [[-0.1725893046286806, 0.26335604636440313, 0.6459083372030877], [-0.6326941400419128, 1.092481289049125, -0.41767020278333805], [0.8824079364924252, -0.860591108800683, -0.840187854098969], [-0.9480281488519483, -0.23824981220954755, 0.1434591685754818], [-0.1226378036824119, -1.4644557001475134, -0.37106982239235536], [0.5926374234227166, -2.1603712585255614, 0.33094948488173087], [2.273636732745319, -0.9182720232453003, 0.18794992960452173], [-2.5702478737951497, -0.5196551220002266, 0.4846497398122398], [-2.8191060547252693, -1.7060396147196757, -0.48067438300388177], [-2.05261409833333, -1.0356953135469942, -0.6200136199693125], [-3.520300958601759, -1.178301322218022, 0.9366997121707941]]
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
png = '/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/HEO/04_sir_viz/D/sir2/sir2_arrows.png'
cmd.png(png, width=W, height=H, dpi=DPI, ray=1)
cmd.quit()
python end
