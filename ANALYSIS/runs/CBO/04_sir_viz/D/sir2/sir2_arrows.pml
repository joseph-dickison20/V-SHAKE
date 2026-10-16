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
cmd.load('/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/CBO/04_sir_viz/D/sir2/sir2_arrows.xyz', 'mol')
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
tips = [[-0.446638215409549, -0.09815128021841331, -0.5783027178116278], [0.48705531581196654, 0.1744666311954152, 2.9391225634139717], [-0.6690222719765646, 0.6004909984185772, 3.5210876267930056], [-0.5459618217742468, 1.9080161646058957, 1.2298544869821122], [0.7997416143612811, 3.5059034917231915, 0.777578051030781], [-0.9933582624989169, 2.365278260415528, 1.721737930039797], [-1.327393751024686, 0.6537709001522919, 0.0072352740968437135], [-1.2638594858833736, 1.7618682937221755, -0.6918323246693465], [-3.623133844085709, 0.8140106218484798, 0.48908067688689577], [-1.108994777020686, -1.2114577378878475, -0.44438267893801164], [-0.4018143492614106, -1.8395385515759992, -2.196009038857424], [1.5450247733661255, 0.29106908836738976, -1.0171675569941356], [-1.3957910332244365, 1.0394476160873538, 2.9037400838617344], [0.09532178862021395, 1.7900336731459598, 4.381951794165406]]
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
png = '/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/CBO/04_sir_viz/D/sir2/sir2_arrows.png'
cmd.png(png, width=W, height=H, dpi=DPI, ray=1)
cmd.quit()
python end
