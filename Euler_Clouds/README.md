# DATABASE INFO

This database is a database of surfaces that equitably represent topology, by designing random surfaces of controlled genus built on top of Lissajous knots randomly.

You can make a database using the class knot_points_cloud.

# DATABASE USE
Use the class knot_points_cloud to visualize and manipulate the elements of this database.

To use this code, you must make sure that Python can execute all the files found in the path `Geometric-Deep-Learning / Random_knots /` (see https://github.com/VMijangos/Geometric-Deep-Learning/tree/main/Random_knots)
```
from points_cloud import knot_points_cloud
load_pcd = knot_points_cloud()
load_pcd.load('knot1')
```

To view the code of class knot_points_cloud, see https://github.com/VMijangos/Geometric-Deep-Learning/blob/main/Random_knots/points_cloud.py
