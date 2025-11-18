from pytspl.cell_complex import CellComplex
from pytspl.io.data_loaders.paper_loader import load_paper_data_cc
from pytspl.simplicial_complex import SimplicialComplex
from pytspl.io.dataset_loader import list_datasets, load_dataset

class TestCellComplexes:
    def test_load_dataset_paper_cc(self):
        data_folder = "pytspl/data/paper_data"

        dataset = "paper"
        cc, coordinates, _ = load_paper_data_cc()

        assert isinstance(cc, CellComplex)
        assert isinstance(coordinates, dict)

        assert cc.nodes == list(coordinates.keys())
        assert len(cc.polygons) == 4
        print("Polygons found:", cc.polygons)

        sc = cc.to_simplicial_complex()
        assert len(sc.triangles) == 3
        print("Triangle: ", sc.triangles)
        # print("\n")
        # print(sc.B1)
        print("\n")
        print(sc.laplacian_matrix())



    def test_convert_sc_to_cc(self):
        data_folder = "pytspl/data/paper_data"

        dataset = "paper"
        sc, coordinates, _ = load_dataset("paper")

        cc = sc.to_cell_complex()

        assert isinstance(cc, CellComplex)
        assert isinstance(coordinates, dict)

        assert cc.nodes == list(coordinates.keys())
        assert len(cc.polygons) == 4
        print("Polygons found:", cc.polygons)
        print("\n")
        print(cc.laplacian_matrix())