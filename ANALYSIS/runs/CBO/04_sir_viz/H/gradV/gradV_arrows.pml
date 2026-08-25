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
cmd.load('/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/CBO/04_sir_viz/H/gradV/gradV_arrows.xyz', 'mol')
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
tips = [[0.3169303912561674, -0.3793259513495953, 1.6899474082180397], [0.26917959916196704, -0.6523312438395169, 0.3840915033768084], [-0.6079622500615234, 1.2773593030543104, 2.767709055454187], [-0.6585170401066043, 1.9582586657130245, 1.4617066955833342], [0.3504353639693365, 2.387600365783369, 1.1855161828856415], [-1.301844654993685, 2.7436280731444778, 1.6517737277974383], [-1.1826918491410574, 1.2057893888591344, 0.1162213138138421], [-1.3957161644413814, 1.801995530589649, -0.6151229312249272], [-2.0892859887750723, 0.6387964412485573, 0.5342939822261907], [-0.46343981926694255, -0.9130085233313364, -0.6032044370369471], [-1.0692091571973332, -1.9363173270681848, -1.7256986771978766], [0.8042843459718625, 0.42015348477002046, -0.5276208463483021], [-1.4834439687981584, 0.7889505033098853, 3.0420350379859302], [-0.11020475972907683, 1.7248266478084378, 3.5348728966076823]]
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
cmd.load_cgo(arrow, 'arrows0')
cmd.set_view(view)
W, H, DPI = 2000, 1400, 600
png = '/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/CBO/04_sir_viz/H/gradV/gradV_arrows.png'
cmd.png(png, width=W, height=H, dpi=DPI, ray=1)
cmd.quit()
python end
