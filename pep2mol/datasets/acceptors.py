import datasets

_CITATION = """\
@article{lopez2017design,
  title={Design principles and top non-fullerene acceptor candidates for organic photovoltaics},
  author={Lopez, Steven A and Sanchez-Lengeling, Benjamin and de Goes Soares, Julio and Aspuru-Guzik, Al{\'a}n},
  journal={Joule},
  volume={1},
  number={4},
  pages={857--870},
  year={2017},
  publisher={Elsevier}
}
"""

_DESCRIPTION = """\
 Non-Fullerene Acceptor Candidates for Organic Photovoltaics
"""

_HOMEPAGE = "https://doi.org/10.1016/j.joule.2017.10.006"

_LICENSE = "Non-commercial use only (https://www.elsevier.com/about/policies/open-access-licenses/elsevier-user-license)"

_URL = "https://ars.els-cdn.com/content/image/1-s2.0-S2542435117301307-mmc2.csv"


# from quantiling ids for 80, 10, 10
_COUNT = 51280
_SPLITS = {
    "train": (0, 80 * _COUNT // 100),
    "dev": (80 * _COUNT // 100, 90 * _COUNT // 100),
    "test": (90 * _COUNT // 100, _COUNT),
}


class AcceptorsDataset(datasets.GeneratorBasedBuilder):
    """Non-Fullerene Acceptor Candidates for Organic Photovoltaics"""

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
                sline = l.split(",")
                cid = int(sline[0])
                if self.config.name == "small" and cid % 100 == 0:
                    continue
                if _SPLITS[split][0] <= cid < _SPLITS[split][1]:
                    yield cid, {"text": sline[2]}
