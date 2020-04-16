#{{{ Import

import gtrace.draw2 as draw
import gtrace.draw2.renderer as renderer

#}}}

#{{{

cv = draw.Canvas()
cv.add_layer('ABC', color=(255,0,255))
cv.add_shape(draw.Line((15,5), (56,-89)), 'ABC')
x = (2.9, 5, 9, 8.0, 20)
y = (-5,-12.0, -3, 6, 18)
cv.add_shape(draw.PolyLine(x, y), 'ABC')
cv.add_shape(draw.Circle((-10,10), 40), 'ABC')
cv.add_shape(draw.Arc((100,50), 40, 30, 200, angle_in_rad=False), 'ABC')
cv.add_shape(draw.Text('This is my test DXF file.',(-50,-50), 5.0, -30, angle_in_rad=False), 'Text')    
renderer.renderDXF(cv, 'Test.dxf')

#}}}
