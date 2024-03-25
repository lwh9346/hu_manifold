import vtk, numpy
from vtkmodules.util.numpy_support import numpy_to_vtk

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(r"outputs\hu_manifold_flow_fix\constraints\interior.vtp")
reader.Update()
pd: vtk.vtkPolyData = reader.GetOutput()
point_data: vtk.vtkPointData = pd.GetPointData()
u, v, w = (
    numpy.array(point_data.GetArray("u")),
    numpy.array(point_data.GetArray("v")),
    numpy.array(point_data.GetArray("w")),
)
velocity = numpy.stack([u, v, w]).transpose()
va = numpy_to_vtk(velocity)
va.SetName("velocity")
point_data.AddArray(va)
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("test.vtp")
writer.SetInputData(pd)
writer.Update()
pass
