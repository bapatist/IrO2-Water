from ase.io import read, write
slab = read("./slab_fixed.xyz")
ads  = read("./adsorbates_baised.xyz", ":")
h2o  = ads[2]
site = slab[44].position
# left
for i, pos in enumerate(h2o.positions):
    h2o.positions[i] = [pos[1], pos[0], pos[2]]
h2o.positions += site+[0,0,2.0]
left = slab+h2o
write("left.xyz", left)
# right
bri = slab[119].position
left[145].position = bri+[-(1.0/2)**0.5, 0, (1.0/2)**0.5]
right = left
write("right.xyz", right)