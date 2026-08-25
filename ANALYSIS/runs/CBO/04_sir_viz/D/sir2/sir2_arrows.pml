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
tips = [[-0.2873441543148486, -0.18472136286152085, -0.23677473753886197], [0.3360265293004775, 0.10421364185117157, 2.084127330075807], [-0.19746828179508846, 1.177985991047811, 2.5508033389933145], [0.16585007185151857, 2.6789876812441387, 2.035125973518707], [-0.37496501049203257, 2.0771546357410746, 1.1293991267493564], [-1.6537307882606995, 2.6501772760526157, 1.6674322276140006], [-0.7897283568851102, 1.499228868929766, 0.8931948178128202], [-1.6334383927191718, 1.7894928365821452, -0.29296507517908277], [-1.7347979345930342, 1.3806309628103999, 0.2412133325219581], [-1.4527055053763398, -1.7995151836734922, -0.5147499005150741], [-0.397533248458106, -2.517477618129785, -2.708320286001899], [0.7980276728402477, 0.2270619010312222, -0.4293827198883159], [-0.826958252402068, 0.9879573815507047, 2.7541110490240035], [-0.8000586686957464, 1.6840311578237477, 3.8704796928132756]]
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
