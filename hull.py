import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from stl import mesh

class Vertex:
    def __init__(self,x,y,z,vnum):
        self.v = [x,y,z]
        self.vnum = vnum
        self.duplicate = None  
        self.onhull = False 
        self.mark = False 
        self.adjedge = None  
        self.next = None
        self.prev = None

class Edge:
    def __init__(self):
        self.adjface = [None, None]
        self.endpts = [None, None]
        self.delete = False  
        self.newface = None 
        self.next = None
        self.prev = None

class Face:
    def __init__(self):
        self.edge = [None, None, None]
        self.vertex = [None, None, None]
        self.visible = False  
        self.prev = None  
        self.next = None 

def MakeVertex(x,y,z,vnum):
    v = Vertex(x,y,z,vnum)
    v.duplicate = None
    v.onhull = False
    v.mark = False
    v.adjedge = None
    return v

edges = []
faces = []

def update(edges):
    for i in range(len(edges)):
        e = edges[i]
        e.next = edges[(i+1)%len(edges)]
        e.prev = edges[(i-1)%len(edges)]

def MakeNullEdge():
    global edges
    e = Edge()
    e.adjface[0] = e.adjface[1] = e.newface = None
    e.endpts[0] = e.endpts[1] = None
    e.delete = False
    edges.append(e)
    return e

def MakeNullFace():
    global faces
    f = Face()
    for i in range(3):
        f.edge[i] = None
        f.vertex[i] = None
    f.visible = False
    faces.append(f)
    return f

def ReadVertices(take_input = False):
    vertices = []
    vnum = 0
 
    if (take_input == True):
        n = int(input())
        for i in range(n):
            x, y, z = map(float, input().split())
            v = MakeVertex(x,y,z,vnum)
            v.v[0] = x
            v.v[1] = y
            v.v[2] = z
            v.vnum = vnum
            vnum += 1
            vertices.append(v)
    else:
        with open('Convex Hull 3D/data.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                x, y, z = map(float, line.split())
                v = MakeVertex(x,y,z,vnum)
                v.v[0] = x
                v.v[1] = y
                v.v[2] = z
                v.vnum = vnum
                vnum += 1
                vertices.append(v)

    for i in range(len(vertices)):
        vertices[i].next = vertices[(i+1)%len(vertices)]
        vertices[i].prev = vertices[(i-1)%(len(vertices))]


    return vertices

vertices = ReadVertices(False)

total_vertices = list(vertices)

def DELETE(list,p):
    list.remove(p)
    update(list)

def Collinear(a,b,c):
    return ((c.v[2]-a.v[2])*(b.v[1]-a.v[1]) - (b.v[2]-a.v[2])*(c.v[1]-a.v[1]) == 0) and ((b.v[2]-a.v[2])*(c.v[0]-a.v[0]) - (b.v[0]-a.v[0])*(c.v[2]-a.v[2]) == 0) and ((b.v[0]-a.v[0])*(c.v[1]-a.v[1]) - (b.v[1]-a.v[1])*(c.v[0]-a.v[0]) == 0) 

def MakeFace(v0, v1, v2, fold):

    e0, e1, e2 = None, None, None

    if not fold:
        e0 = MakeNullEdge()
        e1 = MakeNullEdge()
        e2 = MakeNullEdge()
    else:
        e0 = fold.edge[2]
        e1 = fold.edge[1]
        e2 = fold.edge[0]

    e0.endpts[0] = v0
    e1.endpts[0] = v1
    e2.endpts[0] = v2

    e0.endpts[1] = v1
    e1.endpts[1] = v2
    e2.endpts[1] = v0


    f = MakeNullFace()
    f.edge[0] = e0
    f.edge[1] = e1
    f.edge[2] = e2
    f.vertex[0] = v0
    f.vertex[1] = v1
    f.vertex[2] = v2


    e0.adjface[0] = e1.adjface[0] = e2.adjface[0] = f

    return f

def VolumeSign(f, p):
    ax = f.vertex[0].v[0] - p.v[0]
    ay = f.vertex[0].v[1] - p.v[1]
    az = f.vertex[0].v[2] - p.v[2]
    bx = f.vertex[1].v[0] - p.v[0]
    by = f.vertex[1].v[1] - p.v[1]
    bz = f.vertex[1].v[2] - p.v[2]
    ex = f.vertex[2].v[0] - p.v[0]
    ey = f.vertex[2].v[1] - p.v[1]
    ez = f.vertex[2].v[2] - p.v[2]

    vol = ax * (by * ez - bz * ey) + ay * (bz * ex - bx * ez) + az * (bx * ey - by * ex)

    if vol > 0:
        return 1
    elif vol < -0:
        return -1
    else:
        return 0

def DoubleTriangle(vertices):
    global faces
    global edges
    v0, v1, v2, v3 = None, None, None, None
    f0, f1 = None, None
    e0, e1, e2 = None, None, None


    v0 = vertices[0]
    while Collinear(v0, v0.next, v0.next.next):
        v0 = v0.next
        if (v0 == vertices[0]):
            print("DoubleTriangle: All points are Collinear")
            exit(0)
    v1 = v0.next
    v2 = v1.next


    v0.mark = True
    v1.mark = True
    v2.mark = True


    f0 = MakeFace(v0, v1, v2, f1)
    f1 = MakeFace(v2, v1, v0, f0)


    f0.edge[0].adjface[1] = f1
    f0.edge[1].adjface[1] = f1
    f0.edge[2].adjface[1] = f1
    f1.edge[0].adjface[1] = f0
    f1.edge[1].adjface[1] = f0
    f1.edge[2].adjface[1] = f0


    v3 = v2.next
    vol = VolumeSign(f0, v3)


    while vol == 0:
        v3 = v3.next
        if v3 == v0:
            print('Points are Coplanar')
            exit(0)
        vol = VolumeSign(f0, v3)

    return v3



def MakeCcw(f, e, p):
    fv = None
    i = 0
    s = None
    if e.adjface[0].visible:
        fv = e.adjface[0]
    else:
        fv = e.adjface[1]

    while fv.vertex[i] != e.endpts[0]:
        i += 1


    if fv.vertex[(i + 1) % 3] != e.endpts[1]:
        f.vertex[0] = e.endpts[1]
        f.vertex[1] = e.endpts[0]
    else:
        f.vertex[0] = e.endpts[0]
        f.vertex[1] = e.endpts[1]
        s = f.edge[1]
        f.edge[1] = f.edge[2]
        f.edge[2] = s

    f.vertex[2] = p


def MakeConeFace(e, p):
    new_edge = [None, None]
    new_face = None


    for i in range(2):
        if not e.endpts[i].duplicate:
            new_edge[i] = MakeNullEdge()
            new_edge[i].endpts[0] = e.endpts[i]
            new_edge[i].endpts[1] = p
            e.endpts[i].duplicate = new_edge[i]
        else:
            new_edge[i] = e.endpts[i].duplicate

    new_face = MakeNullFace()
    new_face.edge[0] = e
    new_face.edge[1] = new_edge[0]
    new_face.edge[2] = new_edge[1]
    MakeCcw(new_face, e, p)


    for i in range(2):
        for j in range(2):
            if not new_edge[i].adjface[j]:
                new_edge[i].adjface[j] = new_face
                break
 
    return new_face
    

def AddOne(p):
    global faces
    global edges
    vis = False

    for i in range(len(faces)):
        f = faces[i]
        if VolumeSign(f, p) < 0:
            f.visible = True
            vis = True
        faces[i] = f


    if not vis:
        p.onhull = False
        return False

    for i in range(len(edges)):
        e = edges[i]
        if e.adjface[0].visible and e.adjface[1].visible:
            e.delete = True
        elif e.adjface[0].visible or e.adjface[1].visible:
            e.newface = MakeConeFace(e, p)
        edges[i] = e

    return True




def CleanEdges(edges):
    e = edges[0]  
    t = None 
    for i in range(len(edges)):
        e = edges[i]
        if e.newface:
            if e.adjface[0].visible:
                e.adjface[0] = e.newface
            else:
                e.adjface[1] = e.newface
            e.newface = None
        edges[i] = e

    i = 0
    while i<len(faces):
        e = edges[i]
        if e.delete:
            t = e
            DELETE(edges,t)
        else:
            i+=1



def CleanFaces(faces): 
    t = None  
    i = 0
    while i < len(faces):
        f = faces[i]
        if f.visible:
            DELETE(faces, f)
        else:
            i+= 1

    

def CleanVertices(edges,vertices):
    t = None
    for i in range(len(edges)):
        e = edges[i]
        e.endpts[0].onhull = True
        e.endpts[1].onhull = True
        edges[i] = e

    i = 0

    while i < len(vertices):
        v = vertices[i]
        if v.mark and (v.onhull==False):
            DELETE(vertices,v)
        else:
            i+=1

    for i in range(len(vertices)):
        v = vertices[i]
        v.duplicate = None
        v.onhull = False

def CleanUp(edges, faces, vertices):
    CleanEdges(edges)
    CleanFaces(faces)
    CleanVertices(edges,vertices)

def ConstructHull(vertices, v3, faces, edges):
    v = v3
    for i in range(len(vertices)-3):
        vnext = v.next
        if not v.mark:
            v.mark = True  
            AddOne(v)
            CleanUp(edges,faces,vertices)     
        v = vnext

    
    print([vertex.v for vertex in vertices])



v3 = DoubleTriangle(vertices)

ConstructHull(vertices, v3, faces, edges)
    
i = 0

for face in faces:
    print(f'Face {i+1} = {face.vertex[0].v},{face.vertex[1].v},{face.vertex[2].v} ')
    i+= 1

vertices_list=[]
total_vertices_list=[]
edges_list=[]
faces_list=[]
for vertex in vertices:
    vertices_list.append(vertex.v)
vertices_list = np.array(vertices_list)
for vertex in total_vertices:
    total_vertices_list.append(vertex.v)
total_vertices_list = np.array(total_vertices_list)
for edge in edges:
    edges_list.append([vertices.index(edge.endpts[0]), vertices.index(edge.endpts[1])])
for face in faces:
    faces_list.append([vertices.index(face.vertex[0]), vertices.index(face.vertex[1]), vertices.index(face.vertex[2])])
edges_list = np.array(edges_list)
faces_list = np.array(faces_list)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(total_vertices_list[:, 0], total_vertices_list[:, 1], total_vertices_list[:, 2], c='b', marker='o', label='All Vertices')

ax.scatter(vertices_list[:, 0], vertices_list[:, 1], vertices_list[:, 2], c='y', marker='o', label='On Hull Vertices')
ax.add_collection3d(Poly3DCollection([vertices_list[face] for face in faces_list], alpha=0.5, facecolor='c'))


edge_lines = [vertices_list[edge] for edge in edges_list]
ax.add_collection3d(Line3DCollection(edge_lines, colors='g', linewidths=1, linestyles='solid'))


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.legend()
plt.show()

num_faces = len(faces_list)
mesh_data = mesh.Mesh(np.zeros(num_faces, dtype=mesh.Mesh.dtype))


for i in range(num_faces):
    for j in range(3):
        vertex_index = faces_list[i][j]
        mesh_data.vectors[i][j] = vertices_list[vertex_index]


mesh_data.save('output.stl')
