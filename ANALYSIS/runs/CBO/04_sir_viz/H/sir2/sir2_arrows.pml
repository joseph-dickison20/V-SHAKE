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
cmd.load('/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/CBO/04_sir_viz/H/sir2/sir2_arrows.xyz', 'mol')
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
tips = [[-0.18126152359910902, -0.45703180499762275, 0.9214728570443182], [0.35845908874202126, -0.1346134514712185, 3.234006505738816], [-0.7987087627959489, 1.242404170097219, 2.171308521213491], [-1.1457066777398204, 2.415914397501222, 0.03633456543315461], [0.47118665136829657, 2.9124023341688403, 1.2293956435456639], [-1.2483859088258071, 3.224099473703802, 1.9490611690190267], [-1.610649364102917, 0.9529236829669628, 1.2542094872395173], [-1.2883054101917342, 1.8540831545630525, -0.6554303354549563], [-2.0007563704192473, 0.27386275592409626, 0.2579711639255802], [0.015113541602801428, -1.0708551515778846, -1.1174612794787462], [-1.165831428584369, -1.443881413204327, -1.985003196048206], [0.7258938302210524, -0.04599283050604708, -1.2348599989267295], [-1.430379768284038, 0.7068202442385161, 2.9368994796808625], [0.4550240026088146, 1.315531898593387, 4.047379507068206]]
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
png = '/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/projects/VSHAKE/ANALYSIS/runs/CBO/04_sir_viz/H/sir2/sir2_arrows.png'
cmd.png(png, width=W, height=H, dpi=DPI, ray=1)
cmd.quit()
python end
