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
cmd.load('/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/HEO/04_sir_viz/D/gperp_gradV/gperp_gradV_arrows.xyz', 'mol')
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
tips = [[1.1466690970181759, -0.17457270198997188, -1.000000000000001e-08], [-1.2979288321714835, 1.8225926191195887, 2.9999999999999964e-08], [1.6987960335152226, 1.1721395973790538, -9.99999999999999e-09], [-0.6482649896278677, -2.116055299769319, 2.999999999999999e-08], [-0.9832688531477228, -1.1874428793068976, 1.0000000000000017e-08], [-0.03469457741558547, -2.265667759560087, 1.0000000000000037e-08], [2.234008634950544, -1.094755448628013, -1.999999999999996e-08], [-2.511475368275365, -0.9815324661658626, 8.999999999999999e-08], [-3.1452022975293548, -0.8480682393368197, -0.892483174620455], [-2.4030143548598324, -2.2045063801859, -1.4e-07], [-3.1452020175293547, -0.8480686093368198, 0.892483674620455]]
r, g, b = _rgb('#2ca02c')
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
bases = [[1.1949655599999995, 0.8506201600000001, -1.000000000000001e-08], [-1.21675875, 0.8140536300000001, 2.9999999999999964e-08], [1.1888516199999994, -0.43503535, -9.99999999999999e-09], [-1.21770142, -0.4716057, 2.999999999999999e-08], [0.005419849999999893, -1.17931957, 1.0000000000000017e-08], [0.032802139999999924, -2.26531147, 1.0000000000000037e-08], [2.1685176099999994, -0.93390577, -1.999999999999996e-08], [-2.5517851999999994, -1.1608726, 8.999999999999999e-08], [-3.1230641599999998, -0.84649899, -0.884756], [-2.4477196599999997, -2.25141892, -1.4e-07], [-3.1230638799999997, -0.84649936, 0.8847565]]
tips = [[-0.9155527184621519, 1.1574443470541051, -1.000000000000001e-08], [0.8501338949777622, 1.190909991055269, 2.9999999999999964e-08], [1.286602405814271, -0.49867523058441887, -9.99999999999999e-09], [-1.3162099638438398, -0.4674512989740805, 2.999999999999999e-08], [0.02858011973640697, -1.2838253116342728, 1.0000000000000017e-08], [0.03324735969830642, -2.2490771676233408, 1.0000000000000037e-08], [2.186170128766628, -0.9632696307644657, -1.999999999999996e-08], [-2.5532741797857263, -1.193804115300564, 8.999999999999999e-08], [-3.119941725162208, -0.8375945960338702, -0.8816925346586726], [-2.4474395548917935, -2.251141763366617, -1.4e-07], [-3.119941445162208, -0.8375949660338703, 0.8816930346586725]]
r, g, b = _rgb('#ff7f0e')
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
cmd.load_cgo(arrow, 'arrows1')
cmd.set_view(view)
W, H, DPI = 2000, 1400, 600
png = '/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/HEO/04_sir_viz/D/gperp_gradV/gperp_gradV_arrows.png'
cmd.png(png, width=W, height=H, dpi=DPI, ray=1)
cmd.quit()
python end
