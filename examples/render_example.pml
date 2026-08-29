# Render a Hyper-Fold-Pocket prediction: cartoon protein, predicted pocket
# residues in warm colors, co-crystallized ligand in green (reference only).
# Usage: pymol -cq render_example.pml -- <name> <resi_list>
# e.g.   pymol -cq render_example.pml -- 2rk2A 12+15+16+18+30+47+48+49
import sys
from pymol import cmd

name = sys.argv[1]
resi = sys.argv[2]

cmd.load(f'examples/{name}.pdb', 'protein')
cmd.load(f'examples/{name}_ligand.pdb', 'ligand')
cmd.hide('everything')
cmd.show('cartoon', 'protein')
cmd.color('gray80', 'protein')
cmd.select('pocket', f'protein and resi {resi}')
cmd.show('sticks', 'pocket')
cmd.color('violet', 'pocket')
cmd.show('sticks', 'ligand')
cmd.color('tv_orange', 'ligand')
cmd.set('ray_shadows', 0)
cmd.set('antialias', 2)
cmd.bg_color('white')
cmd.set('ray_opaque_background', 1)
cmd.orient('pocket or ligand')
cmd.zoom('pocket or ligand', 6)
cmd.png(f'examples/{name}_prediction.png', width=1200, height=900, ray=1)
cmd.quit()
