import tornado
import tornado.ioloop
import tornado.web
import os, uuid
import openface
import cv2
import numpy as np

__UPLOADS__ = "uploads/"

class Userform(tornado.web.RequestHandler):
    def get(self):
        self.render("fileuploadform.html")


class Upload(tornado.web.RequestHandler):
    def post(self):
        fileinfo1 = self.request.files['filearg1'][0]
	fileinfo2 = self.request.files['filearg2'][0]
        fname1 = fileinfo1['filename']
	fname2 = fileinfo2['filename']
        extn1 = os.path.splitext(fname1)[1]
        extn2 = os.path.splitext(fname2)[1]
        cname1 = str(uuid.uuid4())+extn1
	cname2 = str(uuid.uuid4())+extn2
        fh1 = open(__UPLOADS__ + cname1, 'w')
        fh2 = open(__UPLOADS__ + cname2, 'w')
        fh1.write(fileinfo1['body'])
        fh2.write(fileinfo2['body'])


	openfaceDir = os.path.join('..', 'openface')
	modelDir = os.path.join(openfaceDir, 'models')
	dlibModelDir = os.path.join(modelDir, 'dlib')
	openfaceModelDir = os.path.join(modelDir, 'openface')
	dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
	networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
	align =openface.AlignDlib(dlibFacePredictor)
	net = openface.TorchNeuralNet(networkModel, 96)

	bgrImg1 = cv2.imread(__UPLOADS__+cname1)
	bgrImg2 = cv2.imread(__UPLOADS__+cname2)

	rgbImg1 = cv2.cvtColor(bgrImg1, cv2.COLOR_BGR2RGB)
	rgbImg2 = cv2.cvtColor(bgrImg2, cv2.COLOR_BGR2RGB)

	bb1 = align.getLargestFaceBoundingBox(rgbImg1)
	bb2 = align.getLargestFaceBoundingBox(rgbImg2)
	
	alignedFace1 = align.align(96, rgbImg1, bb1, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
	alignedFace2 = align.align(96, rgbImg2, bb2, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

	rep1 = net.forward(alignedFace1)
	rep2 = net.forward(alignedFace2)

	d = rep1 - rep2
	D = np.dot(d,d)
        self.finish("<img src="+cname1 +" /><img src="+cname2+"/>"+str(D))


application = tornado.web.Application([
        (r"/", Userform),
        (r"/upload", Upload),
	(r"/(.*)", tornado.web.StaticFileHandler, {'path':'./uploads'})
        ], debug=True)


if __name__ == "__main__":
    application.listen(80)
    tornado.ioloop.IOLoop.instance().start()

