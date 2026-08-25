import numpy as np
import plotly.graph_objects as go

# ============================ USER-CUSTOMIZABLE COLORS ============================
REACTANT_COLOR = '#8e6fb0'   # reactant surface
PRODUCT_COLOR  = '#3fa8a0'   # product surface
MECP_COLOR     = '#c0392b'   # MECP marker fill (e.g. red '#c0392b', blue '#2c5fb0')
SEAM_COLOR     = 'firebrick'   # crossing-seam line color
ARROW_COLOR    = '#000000'   # g_perp vector (shaft + head) color
AXIS_LABEL_COLOR = 'black'   # E, Q1, Q2 text

# g_perp arrowhead shape: smaller HEAD_SIZE = sharper/pointier tip
ARROW_HEAD_SIZE  = 0.2      # cone size (absolute); lower -> sharper (was ~0.20)
ARROW_SHAFT_WIDTH = 7        # shaft thickness

# MECP marker (drawn as a 2D paper-space overlay ON TOP of the whole 3D scene,
# so nothing can occlude it -- the true "bring to front").
MECP_DOT_SIZE     = 8    # overlay dot diameter (px)
MECP_BORDER_WIDTH = 2.5   # black outline thickness (px)
# Fine-tune overlay placement (paper fraction). The 3D->2D projection is reconstructed
# manually from the camera, so a small nudge is usually needed to sit it on the seam.
MECP_OVERLAY_DX   = 0.0   # + moves dot right
MECP_OVERLAY_DY   = 0.005   # + moves dot up

# ---- surface transparency (Option B): lower = more see-through ----
REACTANT_OPACITY = 0.80 #0.88 (original)
PRODUCT_OPACITY  = 0.40 #0.55 (original)

# ============================ STYLE TOGGLES (experiment here) =====================
SURFACE_MESH      = True      # draw wireframe grid lines on the surfaces
MESH_COLOR        = '#000000' # color of the surface mesh lines
MESH_WIDTH        = 1         # thickness of surface mesh lines
MESH_EVERY        = 0.35      # spacing between mesh lines (data units; smaller = denser)
SEAM_WIDTH        = 9         # thickness of crossing seam lines

BACKGROUND_STYLE  = 'none'  # 'linegrid' | 'plotly' | 'none'
#   'linegrid' -> white panes with faint gray gridlines (clean, journal-style)
#   'plotly'   -> default light-blue panes + white gridlines (original look)
#   'none'     -> no panes, no gridlines, no axis lines (floating surfaces)
GRID_COLOR        = '#cccccc'   # gridline color when BACKGROUND_STYLE='linegrid'
# =================================================================================

# ---- camera (defined once so both the scene and the overlay projection use it) ----
CAM_EYE    = dict(x=0.55, y=-1.85, z=0.5)
CAM_CENTER = dict(x=0.0,  y=0.0,   z=0.0)
CAM_UP     = dict(x=0.0,  y=0.0,   z=1.0)
ASPECT     = dict(x=1.0,  y=1.0,   z=0.65)

# ---- grid ----
n = 96                        # even count shifts quad boundaries off the artifact spot
L = 2.2
x = np.linspace(-L, L, n)
y = np.linspace(-L, L, n)
X, Y = np.meshgrid(x, y)

# ---- two diabatic vibronic surfaces ----
a1, a2 = 3.2, 1.9
sep = 1.42
Z0 = 1.2                      # lift whole landscape up off the base plane
Z_R = a1 * (X + sep)**2 + a2 * Y**2 + 0.12 * Y + Z0     # reactant
Z_P = a1 * (X - sep)**2 + a2 * Y**2 - 0.12 * Y + 0.08 + Z0  # product

ZMAX = float(np.nanmax([Z_R, Z_P]))
ZDISP = 0.55 * ZMAX          # fixed z-window (< ZMAX) so stiffer curvature reads as a higher-climbing seam

# ---- crossing seam ----
y_seam = np.linspace(-L, L, 200)
x_seam = (-0.24 * y_seam + 0.08) / (4 * a1 * sep)
z_seam = a1 * (x_seam + sep)**2 + a2 * y_seam**2 + 0.12 * y_seam + Z0

# ---- MECP = true minimum-energy point on the crossing seam ----
i_mecp = int(np.argmin(z_seam))
p_mecp = np.array([x_seam[i_mecp], y_seam[i_mecp], z_seam[i_mecp]])

dx = np.gradient(x_seam, y_seam)
tangent = np.array([dx[i_mecp], 1.0, 0.0]); tangent /= np.linalg.norm(tangent)
g_perp = np.array([tangent[1], -tangent[0], 0.0]); g_perp /= np.linalg.norm(g_perp)
if g_perp[0] < 0:
    g_perp = -g_perp

# ---- surface mesh (wireframe) config ----
def contour_kw():
    if not SURFACE_MESH:
        return dict()
    line = dict(show=True, color=MESH_COLOR, width=MESH_WIDTH,
                start=-L, end=L, size=MESH_EVERY)
    return dict(contours=dict(x=dict(line), y=dict(line)))

# ---- 3D -> 2D paper projection, reconstructing plotly's gl3d camera ----
# Plotly maps scene data into a normalized cube [-1,1]^3 scaled by aspectratio, then
# applies a look-at + perspective projection from 'eye'. We replicate that so we can
# place a 2D overlay marker exactly where the 3D point lands on screen.
def project_to_paper(pt3d):
    # data-space extents actually used by the scene
    xr = (X.min(), X.max())
    yr = (Y.min(), Y.max())
    zr = (0.0, ZDISP)
    # normalize each axis to [-1,1], then scale by aspect (plotly's cube)
    def nrm(v, r, a):
        return (2.0 * (v - r[0]) / (r[1] - r[0]) - 1.0) * a
    P = np.array([nrm(pt3d[0], xr, ASPECT['x']),
                  nrm(pt3d[1], yr, ASPECT['y']),
                  nrm(pt3d[2], zr, ASPECT['z'])])
    eye    = np.array([CAM_EYE['x'],    CAM_EYE['y'],    CAM_EYE['z']]) * 2.0
    center = np.array([CAM_CENTER['x'], CAM_CENTER['y'], CAM_CENTER['z']])
    up     = np.array([CAM_UP['x'],     CAM_UP['y'],     CAM_UP['z']])
    # look-at basis
    f = center - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    # view-space coords
    d = P - eye
    cam = np.array([np.dot(d, s), np.dot(d, u), np.dot(d, f)])
    # perspective divide (fov ~ plotly default). project to NDC-ish plane at z=cam.z
    focal = 1.0 / np.tan(np.radians(43.0) / 2.0)
    ndc_x = focal * cam[0] / cam[2]
    ndc_y = focal * cam[1] / cam[2]
    # map NDC (~[-1,1]) to paper [0,1]; 0.5 center. scale factor tuned for default framing.
    px = 0.5 + 0.5 * ndc_x
    py = 0.5 + 0.5 * ndc_y
    return px, py

# ---- figure ----
fig = go.Figure()

# ---- reactant split into lower/upper halves across the seam (Z_R vs Z_P) ----
eps = 0.15
below = Z_R <= Z_P + eps
Z_R_lower = np.where(below,  Z_R, np.nan)
Z_R_upper = np.where(~below, Z_R, np.nan)

rlight = dict(ambient=1.0, diffuse=0.0, specular=0.0)
rcs = [[0, REACTANT_COLOR], [1, REACTANT_COLOR]]
pcs = [[0, PRODUCT_COLOR],  [1, PRODUCT_COLOR]]
ckw = contour_kw()

# 1) lower reactant (behind)
fig.add_trace(go.Surface(x=X, y=Y, z=Z_R_lower, opacity=REACTANT_OPACITY, showscale=False,
    colorscale=rcs, cmin=0, cmax=ZMAX, lighting=rlight, **ckw))
# 2) product (middle)
fig.add_trace(go.Surface(x=X, y=Y, z=Z_P, opacity=PRODUCT_OPACITY, showscale=False,
    colorscale=pcs, cmin=0, cmax=ZMAX, lighting=rlight, **ckw))
# 3) upper reactant (in front)
fig.add_trace(go.Surface(x=X, y=Y, z=Z_R_upper, opacity=REACTANT_OPACITY, showscale=False,
    colorscale=rcs, cmin=0, cmax=ZMAX, lighting=rlight, **ckw))

# Legend proxy for MECP (invisible in-scene point; the visible dot is the overlay below)
fig.add_trace(go.Scatter3d(x=[p_mecp[0]], y=[p_mecp[1]], z=[p_mecp[2]+0.02],
    mode='markers', marker=dict(size=MECP_DOT_SIZE*0.7, color=MECP_COLOR,
                                line=dict(color='black', width=2)),
    name='MECP', showlegend=True))

fig.add_trace(go.Scatter3d(x=x_seam, y=y_seam, z=z_seam+0.02, mode='lines',
    line=dict(color=SEAM_COLOR, width=SEAM_WIDTH), name='Diabatic Seam', showlegend=True))

alen = 1.15
tip = p_mecp + np.append(g_perp[:2]*alen, 0)
shaft_end = p_mecp + np.append(g_perp[:2]*(alen*0.93), 0)
fig.add_trace(go.Scatter3d(
    x=[p_mecp[0], shaft_end[0]], y=[p_mecp[1], shaft_end[1]],
    z=[p_mecp[2]+0.05, shaft_end[2]+0.05],
    mode='lines', line=dict(color=ARROW_COLOR, width=ARROW_SHAFT_WIDTH), showlegend=False))
fig.add_trace(go.Cone(
    x=[tip[0]], y=[tip[1]], z=[tip[2]+0.05],
    u=[g_perp[0]], v=[g_perp[1]], w=[0.0],
    sizemode='absolute', sizeref=ARROW_HEAD_SIZE, anchor='tip',
    showscale=False, colorscale=[[0, ARROW_COLOR], [1, ARROW_COLOR]]))

def label(pt, text, dz=0.25, color='black', size=20):
    fig.add_trace(go.Scatter3d(x=[pt[0]], y=[pt[1]], z=[pt[2]+dz], mode='text',
        text=[text], textfont=dict(size=size, color=color), showlegend=False))

label([tip[0]+0.0, tip[1]+0.0, tip[2]-1.5], '<b><i>g</i><sub>\u22a5</sub></b>', dz=0.35, color='#111111', size=26)
label([-L*0.75, -L*0.75, a1*(-L*0.72+sep)**2 + a2*(L*0.70)**2 - 0.12*L*0.70 + Z0], '<b>Reactant</b>', dz=3.5, color=REACTANT_COLOR, size=25)
label([ L*0.75,  L*0.75, a1*( L*0.88-sep)**2 + a2*(L*0.88)**2 - 0.12*L*0.88+0.08 + Z0], '<b>Product</b>', dz=3.0, color=PRODUCT_COLOR, size=25)

# ---- axis / background pane styling ----
if BACKGROUND_STYLE == 'linegrid':
    axis_common = dict(showticklabels=False, title=dict(text=''),
                       showbackground=False, showgrid=True,
                       gridcolor=GRID_COLOR, gridwidth=1,
                       zeroline=False, showline=False)
elif BACKGROUND_STYLE == 'none':
    axis_common = dict(showticklabels=False, title=dict(text=''),
                       showbackground=False, showgrid=False,
                       zeroline=False, showline=False, visible=False)
else:  # 'plotly' default look
    axis_common = dict(showticklabels=False, title=dict(text=''))

xaxis = dict(axis_common)
yaxis = dict(axis_common)
zaxis = dict(axis_common); zaxis['range'] = [0, ZDISP]

fig.update_layout(
    scene=dict(
        xaxis=xaxis, yaxis=yaxis, zaxis=zaxis,
        camera=dict(eye=CAM_EYE, center=CAM_CENTER, up=CAM_UP),
        aspectratio=ASPECT),
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=False,
    legend=dict(x=0.10, y=0.05, font=dict(size=22, color='black'),
                bgcolor='rgba(255,255,255,0.6)', bordercolor='black', borderwidth=1,
                itemsizing='constant'),
    width=950, height=750)

# ---- MECP overlay dot: 2D paper-space, drawn ON TOP of the whole 3D scene ----
mx, my = project_to_paper(p_mecp)
mx += MECP_OVERLAY_DX
my += MECP_OVERLAY_DY
fig.add_shape(type='circle', xref='paper', yref='paper',
    x0=mx - MECP_DOT_SIZE/950.0, x1=mx + MECP_DOT_SIZE/950.0,
    y0=my - MECP_DOT_SIZE/750.0, y1=my + MECP_DOT_SIZE/750.0,
    fillcolor=MECP_COLOR, line=dict(color='black', width=MECP_BORDER_WIDTH),
    layer='above')

# upright axis labels (positions manually set; DO NOT CHANGE)
#fig.add_annotation(text='<i>E</i>', x=0.1, y=0.45, xref='paper', yref='paper',
#    showarrow=False, font=dict(size=25, color=AXIS_LABEL_COLOR))
#fig.add_annotation(text='<i>Q\u2081</i>', x=0.40, y=0.12, xref='paper', yref='paper',
#    showarrow=False, font=dict(size=25, color=AXIS_LABEL_COLOR))
#fig.add_annotation(text='<i>Q\u2082</i>', x=0.78, y=0.26, xref='paper', yref='paper',
#    showarrow=False, font=dict(size=25, color=AXIS_LABEL_COLOR))

fig.write_image('seam_schematic.png', scale=3)
print('saved seam_schematic.png ZMAX=%.2f MECP_y=%.2f' % (ZMAX, p_mecp[1]))
print('MECP overlay paper coords: (%.3f, %.3f) -- nudge with MECP_OVERLAY_DX/DY if off' % (mx, my))