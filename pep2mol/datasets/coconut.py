import datasets

_CITATION = """\

@article{sorokina_coconut_2021,
	title = {{COCONUT} online: {Collection} of {Open} {Natural} {Products} database},
	volume = {13},
	issn = {1758-2946},
	url = {https://doi.org/10.1186/s13321-020-00478-9},
	doi = {10.1186/s13321-020-00478-9},
	number = {1},
	journal = {Journal of Cheminformatics},
	author = {Sorokina, Maria and Merseburger, Peter and Rajan, Kohulan and Yirik, Mehmet Aziz and Steinbeck, Christoph},
	month = jan,
	year = {2021},
	pages = {2},
}

"""

_DESCRIPTION = """\
A list of natural products for cheminformatics.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://coconut.naturalproducts.net/"

_LICENSE = "CC-BY-SA 2022"

_URL = "https://coconut.naturalproducts.net/download/smiles"

# from quantiling ids for 80, 10, 10
_COUNT = 407270
_SPLITS = {
    "train": (0, 80 * _COUNT // 100),
    "dev": (80 * _COUNT // 100, 90 * _COUNT // 100),
    "test": (90 * _COUNT // 100, _COUNT),
}


class CoconutDataset(datasets.GeneratorBasedBuilder):
    """A list of unique natural products from Coconut open-source open-data portal."""

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
        header = False
        with open(path, "r") as f:
            for l in f.readlines():
                if header:
                    header = False
                    continue
                sline = l.split()
                cid = int(sline[1][4:]) # SMILES CPN#IDX
                if self.config.name == "small" and cid % 100 == 0:
                    continue
                if _SPLITS[split][0] <= cid < _SPLITS[split][1]:
                    yield cid, {"text": sline[0]}
