import datasets

_CITATION = """\
@article{irwin2020zinc20,
  title={ZINC20—a free ultralarge-scale chemical database for ligand discovery},
  author={Irwin, John J and Tang, Khanh G and Young, Jennifer and Dandarchuluun, Chinzorig and Wong, Benjamin R and Khurelbaatar, Munkhzul and Moroz, Yurii S and Mayfield, John and Sayle, Roger A},
  journal={Journal of chemical information and modeling},
  volume={60},
  number={12},
  pages={6065--6073},
  year={2020},
  publisher={ACS Publications}
}
"""

_DESCRIPTION = """\
A list of commercially available chemical compounds.
"""

_HOMEPAGE = "https://files.docking.org/zinc20-ML/"

_LICENSE = "unknown"

_CHUNKS = 20
_URL = f"https://files.docking.org/zinc20-ML/smiles/ZINC20_smiles_chunk_"


class ZincDataset(datasets.GeneratorBasedBuilder):
    """Purchasable Commercial Molecules from ZINC"""

    VERSION = datasets.Version("20.0.0")
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
        # picked by rolling fair die
        test_chunks = [10]
        val_chunks = [9]
        if self.config.name == "small":
            train_chunks = [0]
        else:
            train_chunks = [
                i
                for i in range(_CHUNKS)
                if i not in val_chunks and i not in test_chunks
            ]
        urls = {
            k: _URL + str(k + 1) + ".tar.gz"
            for k in train_chunks + val_chunks + test_chunks
        }
        paths = dl_manager.download(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepaths": [
                        dl_manager.iter_archive(paths[i]) for i in train_chunks
                    ],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepaths": [
                        dl_manager.iter_archive(paths[i]) for i in test_chunks
                    ],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepaths": [
                        dl_manager.iter_archive(paths[i]) for i in val_chunks
                    ],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepaths, split):
        key = 0

        for chunk in filepaths:
            for path, f in chunk:
                if path.endswith("txt"):
                    for l in f:
                        line = l.decode()
                        yield key, {"text": line.split()[0]}
                        key += 1
                        if self.config.name == "small":
                            if key > 10000:
                                return
