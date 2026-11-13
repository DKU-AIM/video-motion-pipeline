"""Regression checks: margin semantics, image boundary and identity-gap safety."""
import unittest
from crop_geometry import expand_box, segment_boxes

class CropChecks(unittest.TestCase):
    def test_ten_percent_is_each_edge_not_ten_percent_total(self):
        self.assertEqual(expand_box([100,200,100,200],1920,1080), (90,180,210,420))

    def test_boundary_clamped_without_negative_python_slice(self):
        self.assertEqual(expand_box([0,0,100,100],105,105), (0,0,105,105))

    def test_fractional_edges_round_outward(self):
        self.assertEqual(expand_box([10.25,20.25,5,10],100,100), (9,19,16,32))

    def test_missing_target_rejected_instead_of_using_neighbor(self):
        frames=[{'frame':0,'observations':[{'id':9,'bbox_xywh':[1,2,10,20]}]},
                {'frame':1,'observations':[{'id':8,'bbox_xywh':[1,2,10,20]}]}]
        with self.assertRaisesRegex(ValueError,'missing.*1'):
            segment_boxes(frames,9,0,2)

    def test_exact_target_and_end_exclusive(self):
        frames=[{'frame':i,'observations':[{'id':9,'bbox_xywh':[i,2,10,20]},
                                         {'id':8,'bbox_xywh':[99,2,10,20]}]} for i in range(3)]
        self.assertEqual(segment_boxes(frames,9,1,3), [[1,2,10,20],[2,2,10,20]])

if __name__=='__main__': unittest.main()
