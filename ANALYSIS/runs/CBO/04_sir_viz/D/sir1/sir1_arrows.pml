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
cmd.load('/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/CBO/04_sir_viz/D/sir1/sir1_arrows.xyz', 'mol')
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
cmd.load('/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/TRAJS/xDENPLT/OUTPUT/CBO_D.cube', 'dens')
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
bases = [[-0.1119488099999999, 0.00193605999999924, -0.10130564000000009], [0.37352706999999996, -0.0020014099999999813, 2.5748141800000015], [-0.46257745000000006, 1.0771296300000006, 2.7298214300000003], [-0.6242798200000002, 1.9339718400000008, 1.4449683999999998], [0.35411391999999997, 2.379910080000001, 1.1912011399999995], [-1.3241579200000002, 2.766663150000001, 1.6469512799999992], [-1.13214082, 1.11037741, 0.2524432599999995], [-1.34962054, 1.76075605, -0.6146732700000013], [-2.08596929, 0.6345173099999999, 0.5346571199999997], [-0.6213972899999998, -1.0222909700000016, -0.9803223699999999], [-1.0370712899999996, -1.8738716200000025, -1.6673828499999999], [0.79755033, 0.4351827099999991, -0.5463878600000005], [-1.5073768800000003, 0.7810317800000007, 3.036736460000001], [-0.11747553000000022, 1.7718961500000012, 3.5421728900000007]]
tips = [[-0.15905662119693265, 0.17797437171360866, -1.8260896459104832], [0.21868261486834498, 0.45940713679668144, 4.562726146308357], [-0.09257556986616311, 0.9804881722539233, 3.0764114781678367], [-0.6835845256644199, 1.9884694255597066, 0.8976983817801416], [0.2113602826424519, 2.2825262588299036, 1.3412145232930517], [-1.0490225970840992, 2.734680355328204, 1.8078000890316572], [-1.3772353983642296, 0.9993526028325473, 0.022426295004393026], [-1.2996094297615226, 1.7512730708373312, -0.25977748041213694], [-1.9614352259320071, 0.623997256327608, 0.5693775622426742], [-0.3586794372480261, -0.8892428236591714, -1.666139934194998], [-1.1910344799666164, -2.1094226380909116, -1.5972767228734066], [0.49666608739706636, 0.21785206137672783, -0.5268305377757457], [-1.4383354254539011, 0.6820625995651827, 3.0620849557940155], [-0.16496459436994332, 1.8557903203286552, 3.580069059544639]]
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
png = '/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/CBO/04_sir_viz/D/sir1/sir1_arrows.png'
cmd.png(png, width=W, height=H, dpi=DPI, ray=1)
cmd.quit()
python end
