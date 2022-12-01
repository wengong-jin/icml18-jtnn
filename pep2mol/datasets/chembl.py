import datasets

_CITATION = """\
@article{mendez2019chembl,
  title={ChEMBL: towards direct deposition of bioassay data},
  author={Mendez, David and Gaulton, Anna and Bento, A Patr{\'\i}cia and Chambers, Jon and De Veij, Marleen and F{\'e}lix, Eloy and Magari{\~n}os, Mar{\'\i}a Paula and Mosquera, Juan F and Mutowo, Prudence and Nowotka, Micha{\l} and others},
  journal={Nucleic acids research},
  volume={47},
  number={D1},
  pages={D930--D940},
  year={2019},
  publisher={Oxford University Press}
}
"""

_DESCRIPTION = """\
A list of chemical compounds used in biological assays.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://www.ebi.ac.uk/chembl/"

_LICENSE = "CC BY-SA-3.0"

_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_31_chemreps.txt.gz"

# from quantiling ids for 80, 10, 10
_SPLITS = {
    "train": (0, 3792451),
    "dev": (3792451, 4280664),
    "test": (4280664, 4804173 + 1),
}


class ChemblDataset(datasets.GeneratorBasedBuilder):
    """A list of chemical compounds used in biological assays from Chembl."""

    VERSION = datasets.Version("30.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="full", version=VERSION, description="Complete dataset"
        ),
        datasets.BuilderConfig(
            name="small", version=VERSION, description="Small version"
        ),
    ]

    DEFAULT_CONFIG_NAME = "full"

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            # License for the dataset if available
            license=_LICENSE,
            citation=_CITATION,
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        path = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "path": path,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "path": path,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "path": path,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, path, split):
        header = True
        with open(path, "r") as f:
            for l in f.readlines():
                if header:
                    header = False
                    continue
                sline = l.split()
                cid = int(sline[0][6:])
                if self.config.name == "small" and cid % 100 == 0:
                    continue
                if _SPLITS[split][0] <= cid < _SPLITS[split][1]:
                    yield cid, {"text": sline[1]}
