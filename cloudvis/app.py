from cloudvis import CloudVis
from cloudvis.segmenter import Segmenter

segmenter = Segmenter()

def callback(req, res):
    # get requested image
    image = req.getImage('image')

    # create segmentation image
    pred = segmenter.segment(image)
    output = segmenter.colourise(pred)

    res.addImage('segmentation', output)

cloudvis = CloudVis(port=6002)
cloudvis.run(callback)