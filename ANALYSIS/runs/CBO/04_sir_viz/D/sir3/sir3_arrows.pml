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
cmd.load('/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/CBO/04_sir_viz/D/sir3/sir3_arrows.xyz', 'mol')
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
tips = [[0.6162792842568723, -0.09262023043168258, 0.07439113628778776], [0.5266128007094366, 0.05644124587366227, 2.884480969801789], [-1.073463523130071, 1.908853080942437, 2.2694600886955976], [-0.28366965825390034, 1.7833714081689518, 2.0557317521323357], [-0.11061009552874995, 2.5284456346054394, 0.95851124672383], [-1.0002472247307408, 2.30361873597072, 2.149586486815243], [-1.290546557630187, -0.8569424314559106, 0.38913847655012496], [-1.0795291758523025, 1.6367420733024085, -0.8107639345493202], [-2.186071035588399, 1.098809467202643, 0.3856577473370436], [-1.2027015259865674, -0.5336972112830993, -0.8513435295897541], [-1.2937214521342935, -1.802865503278549, -1.9641674671663325], [1.0901118292333558, 0.906422277500757, -0.8619976547704123], [-1.6531502754252796, 1.0063809205534628, 2.7484295990552536], [0.09188229006082643, 1.8122487023287575, 3.6165792526768135]]
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
png = '/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/CBO/04_sir_viz/D/sir3/sir3_arrows.png'
cmd.png(png, width=W, height=H, dpi=DPI, ray=1)
cmd.quit()
python end
