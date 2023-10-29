# Convex-Hull-3D
Construct the convex hull of any given set of 3D points using the Incremental Algorithm. 

`hull.py` consists of the incremental algorithm implementation in python. Detailed algorithm can be referred from the book "COMPUTATIONAL GEOMETRY IN C - JOSEPH O'ROURKE" or from [here](https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf). It's time complexity is $O(n^2)$

To run it on a custom dataset, modify `data.txt` with custom set of vertices in euclidean space in the format `x y z` in each line. Codes prints the list of faces of the hull, plots the hull and also outputs an stl file as below. 

<p align="center">
<img src="https://github.com/berserank/Convex-Hull-3D/blob/main/output.stl" alt="Alt Text" width="700" height="400">
</p>

